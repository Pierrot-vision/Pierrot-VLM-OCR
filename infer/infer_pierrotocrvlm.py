#!/usr/bin/env python
"""PierrotOCRVLM 추론 엔트리포인트 — coarse-to-fine 2단계 문서 파싱.

MinerU2.5 골격 그대로, **체크포인트 하나가 프롬프트만 바꿔** 두 역할을 수행한다:

    ① 레이아웃 패스 : 페이지 축소본(1024 토큰 예산)
                      → "Layout Detection (fine):" → bbox + 클래스 + 읽기순서
    ② 인식 패스     : ①의 bbox 로 **원본 해상도에서** crop (2048 토큰 예산)
                      → 클래스별 프롬프트로 배치 생성
                         Text/List-item/Caption/… → "Text Recognition:"
                         Table                     → "Table Recognition:" (OTSL→HTML)
                         Formula                   → "Formula Recognition:" (LaTeX)
                         Picture                   → 인식 생략(자리표시자)
    ③ 조립         : 읽기순서대로 Markdown / JSON

왜 원본에서 crop 하는가: 레이아웃은 축소본에서 읽지만 좌표가 **0~999 정규값**이라
원본에 그대로 적용된다. 축소본에서 자르면 작은 글자가 이미 뭉개진 뒤라 인식이 망가진다.

왜 배치로 생성하는가: 페이지당 요소가 20개 안팎이라 하나씩 돌리면 느리다. 같은
프롬프트끼리 묶어 한 번에 생성한다(GLM-OCR 도 최대 32개 병렬). 우측 패딩 + 어텐션
마스크로 길이가 달라도 안전하다(모델 generate 가 샘플별 마지막 위치·EOS 를 추적).

단일 패스 모드(--mode page)도 지원한다 — 단순 레이아웃 문서는 페이지 전체를
"Page Recognition:" 한 번으로 읽는 게 빠르고 정확하다.

사용 예:
    python infer/infer_pierrotocrvlm.py --model outputs/pierrotocrvlm_stage2/final \\
        --image page.png --out results/parsed/
    python infer/infer_pierrotocrvlm.py --model ... --image page.png --mode page
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image, ImageDraw

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from eval.metrics_ocr import parse_layout                                     # noqa: E402
from pierrot.models.pierrotocrvlm.processor import collate_encoded            # noqa: E402
from pierrot.models.pierrotocrvlm.weights import load_pretrained              # noqa: E402
from tools.otsl import otsl_to_html                                           # noqa: E402
from tools.pierrotocr_common import (                                         # noqa: E402
    PROMPT_FORMULA, PROMPT_LAYOUT_COARSE, PROMPT_LAYOUT_FINE, PROMPT_LAYOUT_WILD,
    PROMPT_PAGE, PROMPT_TABLE, PROMPT_TEXT,
)

# 레이아웃 라벨 규약 — 이름 → 프롬프트. 학습 때 프롬프트로 규약을 나눠 넣었으므로
# 추론에서도 대상 이미지에 맞는 규약을 골라야 한다(문서면 fine, 사진이면 wild).
_LAYOUT_PROMPTS = {"fine": PROMPT_LAYOUT_FINE,
                   "coarse": PROMPT_LAYOUT_COARSE,
                   "wild": PROMPT_LAYOUT_WILD}

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# 레이아웃 클래스 → 인식 프롬프트. Picture 는 인식하지 않는다(자리표시자만).
_CLASS_PROMPT = {
    "Table":   PROMPT_TABLE,
    "Formula": PROMPT_FORMULA,
    "Picture": None,
}
_DEFAULT_PROMPT = PROMPT_TEXT

# Markdown 조립 규칙(클래스 → 접두/후처리).
_MD_PREFIX = {
    "Title":          "# ",
    "Section-header": "## ",
    "List-item":      "- ",
    "Caption":        "*",          # 이탤릭(닫힘은 아래에서 처리)
}


# ------------------------------------------------------------------ #
# 0~999 정규 좌표 → 원본 픽셀 박스. 여유(pad)를 조금 주고 경계를 클램프한다.
# ------------------------------------------------------------------ #
def denorm_box(box, w: int, h: int, pad: int = 2):
    x0, y0, x1, y1 = box
    return (max(0, int(x0 / 1000 * w) - pad), max(0, int(y0 / 1000 * h) - pad),
            min(w, int(x1 / 1000 * w) + pad), min(h, int(y1 / 1000 * h) + pad))


# ------------------------------------------------------------------ #
# 같은 프롬프트의 crop 들을 배치로 묶어 한 번에 생성한다.
# items = [(index, PIL 이미지, 프롬프트)] → {index: 생성 텍스트}
# ------------------------------------------------------------------ #
@torch.no_grad()
def generate_batch(model, processor, items, max_new_tokens: int, device: str,
                   batch_size: int = 8, stop_on_cycle: int = 0,
                   cycle_repeats: int = 8, meta: dict | None = None,
                   no_repeat_ngram: int = 0) -> dict:
    """items → {idx: text}. `meta` 를 주면 {idx: {생성 토큰 수·EOS·상한도달}} 을 채운다.

    ★ 왜 meta 가 필요한가 — 진단에서 "생성이 잘렸다" 를 `</table>` 미닫힘으로 판정했는데,
      변환기가 malformed OTSL 을 처리하다 못 닫은 경우와 구분되지 않는다. 실제 종료
      토큰과 상한 도달 여부를 생성기에서 직접 받아야 한다.
    """
    out = {}
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        enc = [processor.encode_one(img, prompt) for _, img, prompt in chunk]
        batch = collate_encoded(enc, processor.pad_token_id, max_length=None)
        batch = {k: v.to(device) for k, v in batch.items()}
        gen = model.generate(
            batch["input_ids"], batch.get("pixel_values"), batch.get("image_grid_thw"),
            batch["attention_mask"], max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=processor.eos_token_id, stop_on_cycle=stop_on_cycle,
            cycle_repeats=cycle_repeats, no_repeat_ngram=no_repeat_ngram,
        )
        prompt_len = batch["input_ids"].shape[1]
        for (idx, _img, _p), row in zip(chunk, gen):
            new = row[prompt_len:].tolist()
            text = processor.tokenizer.decode(new, skip_special_tokens=True)
            out[idx] = text.strip()
            if meta is not None:
                eos = processor.eos_token_id
                pad = processor.pad_token_id
                # 패딩·EOS 이전까지가 실제 생성분이다.
                n = len(new)
                for j, tok in enumerate(new):
                    if tok in (eos, pad):
                        n = j
                        break
                # ★ `eos in new` 는 쓰지 않는다 — 짧게 끝나는 Caption·Text crop 까지
                #   전부 False 로 나온다(반환 텐서에 EOS 가 안 담기거나 id 불일치).
                #   실제로 쓸 수 있는 신호는 **생성 길이가 예산에 닿았는가** 뿐이다.
                meta[idx] = {"n_new": n, "budget": max_new_tokens,
                             "stop": "hit_max" if n >= max_new_tokens else "early",
                             "last_tok": int(new[max(0, n - 1)]) if new else None}
    return out


# ------------------------------------------------------------------ #
# 밴드 재검출 — 레이아웃이 통째로 건너뛴 영역만 잘라 한 번 더 검출한다.
#
# ★ 왜 필요한가: 2단 문서에서 레이아웃 패스가 **한쪽 단을 통째로 건너뛰는** 실패가 있다
#   (학회지 p02: 우단 y131~488 두 문단). 이건 임계값 문제가 아니라 생성이 그 박스를
#   내지 않은 것이라, 걸러낸 것을 되살릴 방법이 없다. 그런데 **그 영역만 잘라서 주면
#   같은 모델이 정확히 찾아낸다**(실측: y129~352, y354~494). 그래서 빈 자리를 기하로
#   찾아 그 부분만 다시 묻는다. 통읽기(hybrid-page)와 달리 **좌표가 함께 생기므로**
#   읽기순서 제자리에 넣을 수 있고 표 위치도 흐트러지지 않는다.
# ------------------------------------------------------------------ #
@torch.no_grad()
def relayout_bands(model, processor, image: Image.Image, elements, layout_prompt: str,
                   max_new_tokens: int, device: str, *, min_h: int = 60,
                   iou_th: float = 0.3, min_px: int = 40, report: dict | None = None):
    """미검출 밴드를 재검출해 요소 목록에 끼운 새 목록을 돌려준다."""
    from tools.hybrid_merge import (box_iou, column_uncovered_bands,
                                    insert_by_reading_order)
    bands = column_uncovered_bands(elements, min_h=min_h)
    w, h = image.size
    new_els = []
    for bx0, by0, bx1, by1 in bands:
        px0, py0, px1, py1 = denorm_box((bx0, by0, bx1, by1), w, h, pad=4)
        if px1 - px0 < min_px or py1 - py0 < min_px:
            continue
        txt = generate_batch(model, processor, [(0, image.crop((px0, py0, px1, py1)),
                                                 layout_prompt)],
                             max_new_tokens, device, batch_size=1)[0]
        # crop 기준 0~999 → 페이지 기준 0~999
        sx, sy = (px1 - px0) / w, (py1 - py0) / h
        ox, oy = px0 / w * 1000, py0 / h * 1000
        # ★ 글줄 박스의 가로 폭은 **같은 단에 이미 있는 글줄**을 따라간다.
        #   밴드 폭(x 500~1000)을 그대로 쓰면 페이지 여백까지 덮어 그림이 어색하고,
        #   재검출 박스를 그대로 쓰면 줄 끝이 잘린다. 이웃이 정답을 알고 있다.
        #   머리말·꼬리말은 본문 단과 좌우가 다르므로 폭 기준에서 뺀다(쪽번호가 섞이면
        #   단 경계가 엉뚱하게 넓어진다).
        sib = [e["box"] for e in elements
               if e.get("box") and (e["box"][2] - e["box"][0]) < 600
               and e.get("label") not in ("Page-header", "Page-footer")
               and bx0 <= (e["box"][0] + e["box"][2]) / 2 <= bx1]
        tx0 = max(bx0, min(b[0] for b in sib) - 6) if sib else bx0
        tx1 = min(bx1, max(b[2] for b in sib) + 6) if sib else bx1
        for el in parse_layout(txt):
            b = el["box"]
            nb = [ox + b[0] * sx, oy + b[1] * sy, ox + b[2] * sx, oy + b[3] * sy]
            if nb[2] - nb[0] < 5 or nb[3] - nb[1] < 5:
                continue
            # ★ 재검출 박스는 원본 대비 한 겹 더 축소된 crop 에서 나온 값이라
            #   가장자리가 야박하다. 그대로 crop 하면 **줄 끝 글자가 잘려 나간다**
            #   (실측: "다양한 관점에서 [연구]가 진행되고" 처럼 어절이 사라졌다).
            #   밴드 안쪽으로만 여유를 준다.
            #   글줄 계열은 아예 **단 폭 전체**로 넓힌다 — 여백이 조금 들어와도 인식은
            #   멀쩡하지만, 한 글자라도 잘리면 어절이 통째로 사라진다.
            my = (nb[3] - nb[1]) * 0.01 + 3
            if el["label"] in ("Table", "Picture", "Formula"):
                mx = (nb[2] - nb[0]) * 0.02 + 4
                nb = [max(bx0, nb[0] - mx), max(by0, nb[1] - my),
                      min(bx1, nb[2] + mx), min(by1, nb[3] + my)]
            else:
                nb = [tx0, max(by0, nb[1] - my), tx1, min(by1, nb[3] + my)]
            # 이미 있는 박스와 겹치면 중복이다(밴드 경계에 걸친 요소).
            if any(box_iou(nb, e["box"]) > iou_th for e in elements if e.get("box")):
                continue
            new_els.append({"label": el["label"], "box": nb, "from_band": True})
    if report is not None:
        report.update({"bands": len(bands), "added": len(new_els),
                       "band_boxes": [[round(v) for v in bd] for bd in bands]})
    if not new_els:
        return elements
    return insert_by_reading_order(elements, new_els)


# ------------------------------------------------------------------ #
# 인식 결과를 읽기순서대로 Markdown 으로 조립한다.
# 표는 HTML, 수식은 $$…$$, 그림은 자리표시자.
# ------------------------------------------------------------------ #
def to_markdown(elements) -> str:
    lines = []
    for el in elements:
        cls, text = el["label"], el.get("text", "")
        if cls == "Picture":
            lines.append(f"![{cls}]()")
        elif cls == "Table_html":
            # hybrid 병합이 page 쪽 표(HTML/파이프)를 그대로 넣은 것 — 재변환하면 깨진다.
            lines.append(text)
        elif cls == "Table":
            lines.append(otsl_to_html(text) if text else "")
        elif cls == "Formula":
            lines.append(f"$$\n{text}\n$$" if text else "")
        elif cls in ("Page-header", "Page-footer"):
            continue                                       # 머리말/꼬리말은 본문에서 제외
        elif cls == "Caption":
            lines.append(f"*{text}*" if text else "")
        else:
            lines.append(_MD_PREFIX.get(cls, "") + text)
    return "\n\n".join(l for l in lines if l.strip())


# ------------------------------------------------------------------ #
# 레이아웃 박스를 원본 위에 그려 저장한다(검수용).
# ------------------------------------------------------------------ #
def save_layout_viz(image: Image.Image, elements, path: str) -> None:
    viz = image.convert("RGB").copy()
    dr = ImageDraw.Draw(viz)
    w, h = viz.size
    for i, el in enumerate(elements):
        x0, y0, x1, y1 = denorm_box(el["box"], w, h, pad=0)
        dr.rectangle([x0, y0, x1, y1], outline="#eb6834", width=3)
        dr.text((x0 + 4, max(0, y0 - 14)), f'{i}:{el["label"]}', fill="#eb6834")
    viz.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="PierrotOCRVLM 문서 파싱 추론")
    ap.add_argument("--model", required=True, help="학습 산출물 경로(outputs/.../final)")
    ap.add_argument("--image", required=True, help="페이지 이미지")
    ap.add_argument("--out", default="results/parsed", help="출력 디렉토리")
    ap.add_argument("--mode", choices=("coarse-to-fine", "page"), default="coarse-to-fine",
                    help="2단계 파싱(기본) / 페이지 통읽기")
    ap.add_argument("--layout-prompt", choices=tuple(_LAYOUT_PROMPTS), default="fine",
                    help="레이아웃 라벨 규약(fine=문서 세밀 분할·우리 표준, "
                         "coarse=DocLayNet 묶음, wild=야외 사진 어절 박스)")
    ap.add_argument("--batch-size", type=int, default=8, help="crop 병렬 생성 배치")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--layout-max-new-tokens", type=int, default=768)
    # ★ 표만 상한을 따로 준다 — 통계표는 셀이 수백 개라 본문용 상한으로는
    #   아래쪽 행이 통째로 잘린다(KDoc 실측: 표 65개 중 8개가 2048 에서 잘림).
    ap.add_argument("--table-max-new-tokens", type=int, default=4096)
    # ★ 퇴행 반복 차단 기본 ON. 그리디 디코딩이 조밀한 표에서 같은 조각을 상한까지
    #   되풀이한다. 점수를 올리진 않지만(9쪽 ablation 실측 동일) 출력이 절반으로
    #   줄고 생성이 빨라진다 — 반복 뒤 토큰은 어차피 전부 오답이다.
    ap.add_argument("--stop-on-cycle", type=int, default=32, help="0 이면 비활성")
    ap.add_argument("--cycle-repeats", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=tuple(_DTYPES))
    ap.add_argument("--save-viz", action="store_true", help="레이아웃 박스 시각화 저장")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    model, processor = load_pretrained(a.model, device=a.device, dtype=_DTYPES[a.dtype])
    model.eval()
    image = Image.open(a.image).convert("RGB")
    t0 = time.time()

    # ── 단일 패스: 페이지 전체를 한 번에 읽는다 ──
    if a.mode == "page":
        text = generate_batch(model, processor, [(0, image, PROMPT_PAGE)],
                              a.max_new_tokens, a.device, batch_size=1,
                              stop_on_cycle=a.stop_on_cycle,
                              cycle_repeats=a.cycle_repeats)[0]
        md_path = os.path.join(a.out, f"{stem}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[infer] page 모드 완료 {time.time() - t0:.1f}s → {md_path}")
        return

    # ── ① 레이아웃 패스 ──
    layout_prompt = _LAYOUT_PROMPTS[a.layout_prompt]
    layout_text = generate_batch(model, processor, [(0, image, layout_prompt)],
                                 a.layout_max_new_tokens, a.device, batch_size=1)[0]
    elements = parse_layout(layout_text)                   # 나열 순서 = 읽기 순서
    t_layout = time.time() - t0
    print(f"[infer] 레이아웃 {len(elements)}개 요소 ({t_layout:.1f}s)")

    # ── ② 인식 패스: 원본에서 crop → 프롬프트별 배치 생성 ──
    w, h = image.size
    jobs = []
    for i, el in enumerate(elements):
        prompt = _CLASS_PROMPT.get(el["label"], _DEFAULT_PROMPT)
        if prompt is None:                                 # Picture 등은 인식 생략
            continue
        x0, y0, x1, y1 = denorm_box(el["box"], w, h)
        if x1 - x0 < 8 or y1 - y0 < 8:                     # 극소 박스는 건너뜀
            continue
        jobs.append((i, image.crop((x0, y0, x1, y1)), prompt))

    # 같은 프롬프트끼리 묶어야 배치 내 태스크가 섞이지 않는다.
    texts = {}
    for prompt in sorted({p for _, _, p in jobs}):
        group = [j for j in jobs if j[2] == prompt]
        budget = a.table_max_new_tokens if prompt == PROMPT_TABLE else a.max_new_tokens
        texts.update(generate_batch(model, processor, group, budget,
                                    a.device, batch_size=a.batch_size,
                                    stop_on_cycle=a.stop_on_cycle,
                                    cycle_repeats=a.cycle_repeats))
        print(f"[infer]   {prompt:24s} {len(group):3d}개 인식", flush=True)
    for i, el in enumerate(elements):
        el["text"] = texts.get(i, "")

    # ── ③ 조립 ──
    md = to_markdown(elements)
    md_path = os.path.join(a.out, f"{stem}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    json_path = os.path.join(a.out, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"image": os.path.abspath(a.image), "size": [w, h],
                   "elements": elements}, f, ensure_ascii=False, indent=2)
    if a.save_viz:
        save_layout_viz(image, elements, os.path.join(a.out, f"{stem}_layout.jpg"))

    print(f"[infer] 완료 {time.time() - t0:.1f}s "
          f"(레이아웃 {t_layout:.1f}s + 인식 {time.time() - t0 - t_layout:.1f}s)")
    print(f"[infer] → {md_path}\n[infer] → {json_path}")


if __name__ == "__main__":
    main()
