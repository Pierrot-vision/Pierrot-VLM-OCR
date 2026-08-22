#!/usr/bin/env python
"""페이지 이미지 여러 장 → 페이지별 마크다운 (벤치마크 공용 배치 러너).

`infer/infer_pierrotocrvlm.py` 는 이미지 **한 장**용 CLI 라, 849~1,651장을 돌리려면
모델을 매번 다시 올려야 한다(장당 20초 낭비). 여기서는 모델을 한 번만 올리고
이미지 목록을 훑는다. 추론 로직은 그 파일의 함수를 그대로 가져다 쓴다 — 벤치마크
때문에 파싱 규칙이 갈라지면 점수가 실제 제품과 달라진다.

KDoc-OCRBench-V2 도 OmniDocBench end-to-end 도 요구 형식이 "페이지별 마크다운"이라
같은 러너로 둘 다 처리한다.

사용:
    python benchmark/run_pages.py --model outputs/pierrotocrvlm_stage3/final \\
        --images "<PAGES_DIR>/*.jpg" --out results/kdoc_md/stage3
    # GPU 를 나눠 쓰려면 --shard i --num-shards n 으로 쪼개 여러 프로세스로 띄운다.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch                                                          # noqa: E402
from PIL import Image                                                 # noqa: E402

from eval.metrics_ocr import parse_layout                             # noqa: E402
from infer.infer_pierrotocrvlm import (                               # noqa: E402
    _CLASS_PROMPT, _DEFAULT_PROMPT, _LAYOUT_PROMPTS,
    denorm_box, generate_batch, relayout_bands, to_markdown,
)
from pierrot.models.pierrotocrvlm.weights import load_pretrained      # noqa: E402
from tools.hybrid_merge import merge_hybrid, merge_page_base           # noqa: E402
from tools.pierrotocr_common import PROMPT_PAGE, PROMPT_TABLE         # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, help="glob 패턴")
    ap.add_argument("--out", required=True)
    # hybrid = c2f 골격 + page 보충. 페이지당 생성 1회가 늘지만(page 패스) 실패한 표
    # 자리가 빈 문자열로 남는 걸 막는다 — 그 자리는 지금 자동 0점이다.
    # hybrid       = c2f 골격 + page 보충(어제 만든 것). 긴 텍스트 +3.4%p.
    # hybrid-page  = **page 골격** + c2f 표·머리말 이식. 같은 체크포인트 실측에서 page 가
    #                긴 텍스트에 6.0%p 앞서므로(66.3 vs 60.3) 골격을 뒤집은 것.
    ap.add_argument("--mode", choices=("coarse-to-fine", "page", "hybrid", "hybrid-page"),
                    default="coarse-to-fine")
    # hybrid 세부: 밴드 보충을 끄면 "표 교체만" 의 효과를 분리해 볼 수 있다.
    ap.add_argument("--no-band-fill", action="store_true",
                    help="hybrid 에서 레이아웃 미덮 영역 보충을 하지 않는다(표 교체만)")
    # strict = 잘림·격자깨짐·파싱실패만 교체(기본), all = 셀 부족까지 교체.
    # 실측 근거로 기본을 strict 로 둔다 — 셀 부족 136건 중엔 정상적인 2×2 표가 섞여 있어
    # 교체하면 오히려 깎을 수 있다. A/B 로 확인하기 전까지 보수적으로 간다.
    ap.add_argument("--hybrid-replace", choices=("strict", "all"), default="strict")
    # hybrid-page 세부. ★ KDoc 세 테스트는 모두 **위치를 보지 않는다**(present-anywhere /
    #   any-table / absent). 그래서 기본은 근거가 강할 때만 교체하고 나머지는 끝에 붙인다 —
    #   평문 교체는 점수 이득이 0 인데 본문을 훼손할 위험만 있다.
    ap.add_argument("--hybrid-place", choices=("append", "replace", "anchor"),
                    default="replace",
                    help="append=교체 없음(본문 무손실) / replace=표꼴 블록만 / anchor=산문까지")
    ap.add_argument("--keep-hf", action="store_true",
                    help="page 출력의 머리말·꼬리말을 지우지 않는다(효과 분리용)")
    # ★ 표 crop 만 비율 패딩을 준다. KDoc 실표 crop A/B 실측(300개 · 셀 recall):
    #     pad 0% 43.8 / **pad 2% 44.8** / pad 4% 39.9  → 2% 가 최적(4% 는 주변 침범)
    #   상·좌를 더 준다 — KDoc 테스트의 78% 가 left_heading 을 묻는데 헤더가 경계에
    #   붙어 잘리면 그 관계가 통째로 깨진다.
    ap.add_argument("--table-pad", type=float, default=0.0,
                    help="표 crop 비율 패딩(0.02 = 2%%). 다른 클래스는 영향 없음")
    # ★ 표에만 cycle 을 끌 수 있게 한다. 같은 실측에서 cycle 32 → 0 이 셀 recall 을
    #   41.9 → 44.8% 로 올렸다(오절단 58건 소멸). 페이지 통읽기는 반복 폭주 위험이
    #   있으므로 기본값을 유지하고 **표만** 분리한다.
    ap.add_argument("--table-stop-on-cycle", type=int, default=-1,
                    help="표 crop 전용 cycle 상한(-1 = --stop-on-cycle 을 그대로 사용)")
    ap.add_argument("--allow-position-insert", action="store_true",
                    help="anchor 실패 시 상대 y 위치로 삽입(기본 off — 근거 약함)")
    # ★ hybrid-page 는 표 crop 만 이식하고 본문 crop(Text/List-item/Section-header/
    #   Caption/Footnote/Title)을 전량 버린다. page 통읽기가 반복 루프로 무너진 쪽에서는
    #   잘 읽어 둔 crop 까지 함께 사라진다(V3.5 §4.5 — 본문 실패의 34.4%가 이 조립 원인).
    #   849쪽 재조립 실측: Long Text 62.93 → 74.74, 3축 67.44 → 71.29 (+3.85p,
    #   CI [+3.23, +4.52]). 표 42.42 불변. Page-header/footer 는 자동 제외한다(넣으면 −28p).
    #   대가는 읽기순서 훼손이라 **기본 off** 다 — 벤치 경로에서만 켠다.
    ap.add_argument("--body-fill", action="store_true",
                    help="hybrid-page 에서 page 골격이 못 덮은 본문 crop 을 끝에 붙인다")
    ap.add_argument("--layout-prompt", default="fine", choices=tuple(_LAYOUT_PROMPTS))
    # ★ 레이아웃이 통째로 건너뛴 영역만 잘라 재검출한다(2단 문서의 단 누락 대책).
    ap.add_argument("--relayout-bands", action="store_true",
                    help="미검출 밴드를 잘라 레이아웃을 한 번 더 돌린다")
    ap.add_argument("--band-min-h", type=int, default=60,
                    help="이 높이(0~1000) 이상 비어 있어야 재검출 대상")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--layout-max-new-tokens", type=int, default=1024)
    # ★ 표는 따로 더 준다. 실측: 예측 표 65개 중 8개(12%)가 2048 에 정확히 걸려
    #   잘렸다(토큰 수가 2046~2130 에 군집). 통계연보급 표는 셀이 수백 개라
    #   본문용 상한으로는 아래쪽 행이 통째로 사라진다.
    ap.add_argument("--table-max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    # 퇴행 반복 차단(주기 ≤ N 이 cycle-repeats 회 이어지면 종료). 0 이면 비활성.
    # ★ 기본 ON(32). 9쪽 ablation 실측: 출력 83,139자 → 40,252자(절반 이하)로 줄고
    #   생성 시간도 짧아지는데 **점수는 정확히 동일**하다(micro 7.3% 불변).
    #   즉 점수를 올리는 장치가 아니라 쓰레기 출력을 끊는 위생 장치다 — 반복이
    #   시작된 뒤의 토큰은 어차피 전부 오답이라 잃을 것이 없다.
    # ★ 페이지 통읽기 **전용** n-gram 금지. 표·crop 에는 걸지 않는다 — 표의 정당한
    #   반복(빈 셀·같은 숫자·단위 표기)까지 막아 표를 망가뜨린다.
    #   849쪽 실측: page 통읽기의 36.8%(287쪽)가 20자×8회 반복에 걸리고, 그 쪽의 본문
    #   통과율이 48.9% 로 정상(69.6%)보다 20.7p 낮다. body_fill 이 절반만 메워
    #   (격차 20.7p → 14.4p) **+3.86p 가 남아 있다.** `--stop-on-cycle` 은 이미 시작된
    #   반복을 끊을 뿐이라 그 뒤 페이지가 안 읽힌다 — 여기서는 애초에 안 만들게 한다.
    ap.add_argument("--page-no-repeat-ngram", type=int, default=0,
                    help="페이지 통읽기에서 같은 n-gram 재출현 금지(0=끔, 권장 12~24)")
    ap.add_argument("--stop-on-cycle", type=int, default=32)
    # 몇 번 이어져야 루프로 볼지. 4 는 표의 빈 셀 반복까지 끊어 내용을 날렸다
    # (실측: 표 통과율 4.9%→4.4%). 실제 루프는 60회 이상이라 8 이면 충분하다.
    ap.add_argument("--cycle-repeats", type=int, default=8)
    # ★ 표 전용 반복 임계. 공용 `--cycle-repeats 8` 은 표의 **정당한** 반복(빈 셀·같은
    #   숫자)까지 끊어 내용을 날렸다(실측 4.9%→4.4%). 그래서 tuned 하네스는 표의 사이클
    #   차단을 아예 껐는데(`--table-stop-on-cycle 0`), 그 대가로 폭주를 막을 장치가
    #   없어졌다 — 표 crop 13.3% 가 상한에 닿고 그중 46.5% 가 격자 붕괴다(셀 876~1,227개
    #   생성, 학습 표 중앙은 90개). 예산을 6,144 로 늘려도 상한 도달 98.8% 로 그대로다.
    #   즉 예산 부족이 아니라 **종료 실패**다. 표만 더 관대한 임계로 다시 켠다.
    ap.add_argument("--table-cycle-repeats", type=int, default=-1,
                    help="표 crop 의 cycle_repeats(기본 -1 = --cycle-repeats 와 동일)")
    # 레이아웃 원시 출력·bbox 를 함께 저장한다. 이게 없으면 "표를 못 찾은 것"과
    # "찾았는데 못 읽은 것"을 사후에 구분할 수 없다(실제로 구분 못 해 헤맸다).
    ap.add_argument("--save-layout", action="store_true")
    # ★ 진단 전용: crop 원시 출력을 **자르지 않고** 저장하고 생성 메타까지 남긴다.
    #   기존 --save-layout 은 texts 를 2,000자로 잘라 저장했는데, 그걸 원본으로 착각해
    #   "raw 에 셀값 없음 38.7%" 라는 틀린 진단을 했다(실제 16.2%p 가 저장 절단이었다).
    ap.add_argument("--save-trace", action="store_true",
                    help="crop 별 전체 raw OTSL·토큰수·EOS·상한도달·변환 HTML 을 저장")
    # 인식 패스 해상도 상한(픽셀). 기본은 체크포인트에 저장된 값(2MP).
    # ★ 학습·추론 해상도는 원칙적으로 같아야 하지만, "해상도를 올리면 조밀한
    #   표가 읽히는가"를 재학습 없이 먼저 확인하려는 ablation 용이다.
    ap.add_argument("--max-pixels", type=int, default=0)
    return ap.parse_args()


def one_page(model, processor, image, a, diag: dict | None = None) -> str:
    """이미지 한 장 → 마크다운. infer 스크립트의 2단계 흐름과 동일하다."""
    if a.mode == "page":
        return generate_batch(model, processor, [(0, image, PROMPT_PAGE)],
                              a.max_new_tokens, a.device, batch_size=1,
                              stop_on_cycle=a.stop_on_cycle,
                              cycle_repeats=a.cycle_repeats)[0]

    layout_text = generate_batch(model, processor,
                                 [(0, image, _LAYOUT_PROMPTS[a.layout_prompt])],
                                 a.layout_max_new_tokens, a.device, batch_size=1)[0]
    elements = parse_layout(layout_text)                 # 나열 순서 = 읽기 순서
    if getattr(a, "relayout_bands", False) and elements:
        rep = {}
        elements = relayout_bands(model, processor, image, elements,
                                  _LAYOUT_PROMPTS[a.layout_prompt],
                                  a.layout_max_new_tokens, a.device,
                                  min_h=a.band_min_h, report=rep)
        if diag is not None:
            diag["relayout"] = rep
    if diag is not None:
        diag["layout_raw"] = layout_text[:4000]
        diag["elements"] = [{"label": e["label"], "box": e["box"]} for e in elements]
    if not elements:
        # 레이아웃이 비면 페이지 통읽기로 물러선다 — 빈 마크다운보다 낫다.
        return generate_batch(model, processor, [(0, image, PROMPT_PAGE)],
                              a.max_new_tokens, a.device, batch_size=1,
                              stop_on_cycle=a.stop_on_cycle,
                              cycle_repeats=a.cycle_repeats)[0]

    w, h = image.size
    jobs = []
    for i, el in enumerate(elements):
        prompt = _CLASS_PROMPT.get(el["label"], _DEFAULT_PROMPT)
        if prompt is None:                               # Picture 등은 인식 생략
            continue
        x0, y0, x1, y1 = denorm_box(el["box"], w, h)
        if prompt == PROMPT_TABLE and a.table_pad > 0:
            dw, dh = (x1 - x0) * a.table_pad, (y1 - y0) * a.table_pad
            x0 = max(0, int(x0 - dw * 1.5))            # 좌측 헤더를 더 넉넉히
            y0 = max(0, int(y0 - dh * 1.5))            # 상단 헤더도
            x1 = min(w, int(x1 + dw))
            y1 = min(h, int(y1 + dh))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        jobs.append((i, image.crop((x0, y0, x1, y1)), prompt))

    texts, gmeta = {}, {}
    for prompt in sorted({p for _, _, p in jobs}):        # 프롬프트별로 묶어 배치
        group = [j for j in jobs if j[2] == prompt]
        budget = a.table_max_new_tokens if prompt == PROMPT_TABLE else a.max_new_tokens
        cyc = (a.table_stop_on_cycle if (prompt == PROMPT_TABLE and a.table_stop_on_cycle >= 0)
               else a.stop_on_cycle)
        rep = (a.table_cycle_repeats if (prompt == PROMPT_TABLE and a.table_cycle_repeats > 0)
               else a.cycle_repeats)
        texts.update(generate_batch(model, processor, group, budget,
                                    a.device, batch_size=a.batch_size,
                                    stop_on_cycle=cyc,
                                    cycle_repeats=rep, meta=gmeta))
    for i, el in enumerate(elements):
        el["text"] = texts.get(i, "")
    if diag is not None:
        diag["texts"] = {str(i): texts.get(i, "")[:2000] for i in range(len(elements))}
    if diag is not None and getattr(a, "save_trace", False):
        import hashlib
        from infer.infer_pierrotocrvlm import otsl_to_html
        crops = {i: c for i, c, _ in jobs}
        tr = []
        for i, el in enumerate(elements):
            if i not in texts:
                continue
            raw = texts[i]
            m = gmeta.get(i, {})
            rec = {"i": i, "label": el["label"], "box": el["box"],
                   "crop_wh": list(crops[i].size) if i in crops else None,
                   "raw": raw, "raw_chars": len(raw),          # ★ 자르지 않는다
                   "n_new": m.get("n_new"), "budget": m.get("budget"),
                   "stop": m.get("stop"),
                   "ends_nl": raw.rstrip().endswith("<nl>"),
                   "raw_md5": hashlib.md5(raw.encode()).hexdigest()[:12]}
            if el["label"] == "Table":
                html = otsl_to_html(raw) if raw else ""
                rec["html"] = html
                rec["html_md5"] = hashlib.md5(html.encode()).hexdigest()[:12]
            tr.append(rec)
        diag["trace"] = tr

    # ── hybrid-page: page 통읽기를 골격으로 두고 c2f 에서 표·머리말만 가져온다 ──
    if a.mode == "hybrid-page":
        page_md = generate_batch(model, processor, [(0, image, PROMPT_PAGE)],
                                 a.max_new_tokens, a.device, batch_size=1,
                                 stop_on_cycle=a.stop_on_cycle,
                                 cycle_repeats=a.cycle_repeats,
                                 no_repeat_ngram=a.page_no_repeat_ngram)[0]
        md, rep = merge_page_base(elements, page_md, replace_level=a.hybrid_replace,
                                  page_w=w, page_h=h, place=a.hybrid_place,
                                  drop_hf=not a.keep_hf,
                                  allow_position_insert=a.allow_position_insert,
                                  body_fill=a.body_fill)
        if diag is not None:
            diag["hybrid_page"] = rep
            diag["page_md"] = page_md if getattr(a, "save_trace", False) else page_md[:4000]
        return md

    # ── hybrid: page 패스를 한 번 더 돌려 실패한 표·미덮 영역만 보충한다 ──
    # 골격은 c2f 다(좌표와 읽기순서를 가진 쪽). page 는 재료로만 쓴다.
    if a.mode == "hybrid":
        page_md = generate_batch(model, processor, [(0, image, PROMPT_PAGE)],
                                 a.max_new_tokens, a.device, batch_size=1,
                                 stop_on_cycle=a.stop_on_cycle,
                                 cycle_repeats=a.cycle_repeats,
                                 no_repeat_ngram=a.page_no_repeat_ngram)[0]
        elements, rep = merge_hybrid(elements, page_md,
                                     fill_bands=not a.no_band_fill,
                                     replace_level=a.hybrid_replace,
                                     page_w=w, page_h=h)
        if diag is not None:
            diag["hybrid"] = rep
            diag["page_md"] = page_md if getattr(a, "save_trace", False) else page_md[:4000]
    return to_markdown(elements)


def main() -> None:
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(a.images))[a.shard::a.num_shards]
    print(f"[pages] 샤드 {a.shard}/{a.num_shards} — {len(paths):,}장", flush=True)

    model, processor = load_pretrained(a.model, device=a.device, dtype=torch.bfloat16)
    model.eval()
    if a.max_pixels:
        old = processor.max_pixels
        processor.max_pixels = a.max_pixels                  # 레이아웃 예산은 건드리지 않는다
        print(f"[pages] 인식 해상도 상한 {old:,} → {processor.max_pixels:,}", flush=True)

    t0, done, errs = time.time(), 0, 0
    for p in paths:
        dst = os.path.join(a.out, os.path.splitext(os.path.basename(p))[0] + ".md")
        if os.path.exists(dst) and not a.overwrite:
            done += 1
            continue
        diag = {} if a.save_layout else None
        try:
            with torch.no_grad():
                md = one_page(model, processor, Image.open(p).convert("RGB"), a, diag)
        except Exception as e:                            # noqa: BLE001 — 한 장 실패로 멈추지 않는다
            print(f"[pages] 실패 {os.path.basename(p)}: {type(e).__name__}: {e}", flush=True)
            md, errs = "", errs + 1
        with open(dst, "w", encoding="utf-8") as f:
            f.write(md)
        if diag is not None:
            import json as _json
            with open(dst[:-3] + ".layout.json", "w", encoding="utf-8") as f:
                _json.dump(diag, f, ensure_ascii=False)
        done += 1
        if done % 20 == 0:
            el = time.time() - t0
            print(f"[pages] {done}/{len(paths)} ({el / max(1, done):.1f}s/장, 실패 {errs})",
                  flush=True)
    print(f"[pages] 완료 {done:,} / 실패 {errs} / {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
