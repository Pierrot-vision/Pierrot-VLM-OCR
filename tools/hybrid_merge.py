#!/usr/bin/env python
"""page ↔ coarse-to-fine 병합 — 표만 정확히 갈아 끼우고, 못 덮은 영역만 보충한다.

**왜 필요한가.** 두 파이프라인의 강점이 정확히 어긋난다(v3 학습 전 baseline, KDoc 51쪽):

    Stage 1B  표 27.7%  문장 33.7%      ← c2f 가 표에 강하다
    Stage 3   표 26.7%  문장 60.3%      ← page 통읽기가 본문에 강하다

그런데 지금 c2f 조립기는 표 인식이 실패하면 **빈 문자열**을 넣는다
([infer_pierrotocrvlm.to_markdown](../infer/infer_pierrotocrvlm.py#L120) 의
`otsl_to_html(text) if text else ""`). 검출은 맞았는데 crop 생성이 잘리거나 반복으로
끊기면 그 자리가 통째로 사라져 **자동 0점**이다. 병합의 실질 이득은 fallback 이 아니라
여기다 — 실패한 표 자리에 page 결과를 넣으면 0점이 부분점수가 된다.

★ **좌표가 있는 쪽이 골격이다.** c2f 는 모든 요소의 박스(0~999 정규값)와 읽기순서를
  갖고 page 출력은 좌표 없는 평문이다. 그래서 "page 를 기준으로 표 영역을 교체" 하려면
  문자열 anchor 추측이 필요하지만, **거꾸로 c2f 를 기준으로 두면 기하로 결정**된다.
  이 모듈은 후자를 택한다 — 추측이 들어가는 지점을 표 대체 1곳으로 좁힌다.

★ **레이아웃 박스에는 신뢰도 점수가 없다.** 레이아웃이 모델의 텍스트 생성
  (`Table: 119,204,196,228`)이라 확률이 없다. 그래서 임계값 대신 **출력 자체의 검증**을
  쓴다 — OTSL 파싱 가능 / 행 길이 일관 / 셀 수 하한 / 잘림·반복 흔적.

절대 규칙(순서대로 강하다):
  1. c2f 가 만든 내용을 **지우지 않는다**. 병합은 추가·교체만 한다.
  2. 표 주변 제목·주석·출처(Caption/Footnote)는 손대지 않는다.
  3. 이미 정상인 표는 건드리지 않는다(중복 삽입 금지).
  4. 보충은 **읽기순서 위치**에 넣는다 — 문서 끝에 몰아넣지 않는다.

torch 를 import 하지 않는다 — GPU 없이 테스트할 수 있어야 한다.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from tools.otsl import CHED, ECEL, FCEL, LCEL, NL, UCEL, XCEL, otsl_to_html

# 표로 보이는 page 블록 — HTML 표이거나 파이프 표이거나 OTSL 그대로.
_TABLE_HINT = re.compile(r"<table|\|\s*[^|\n]+\s*\||<fcel>|<ched>")
_WS = re.compile(r"\s+")
_MD_ORNAMENT = re.compile(r"^[#>*\-\s|:]+|[*_`]+")

# 셀 수가 이보다 적으면 "표를 읽었다" 고 보지 않는다. 2×2 는 장식용 테두리일 때가 많다.
MIN_CELLS = 4
# 행 길이가 이 비율 미만으로 일치하면 격자가 깨진 것으로 본다.
MIN_ROW_CONSISTENCY = 0.6
# page 블록이 c2f 텍스트와 이 비율 이상 겹치면 중복으로 보고 넣지 않는다.
DUP_OVERLAP = 0.75
# 이 높이(0~1000 기준) 이상 비어 있으면 "못 덮은 밴드" 로 본다. 페이지의 8%.
MIN_BAND_H = 80


def norm_text(s: str) -> str:
    """비교 전용 정규화 — 공백·마크다운 장식·대소문자를 지운다."""
    s = _MD_ORNAMENT.sub("", s or "")
    return _WS.sub("", s).lower()


def overlap_ratio(short: str, long: str) -> float:
    """short 의 문자 중 long 에 들어 있는 비율(3-gram 기준). 중복 판정용."""
    a, b = norm_text(short), norm_text(long)
    if len(a) < 3:
        return 1.0 if a and a in b else 0.0
    grams = {a[i:i + 3] for i in range(len(a) - 2)}
    hit = sum(1 for g in grams if g in b)
    return hit / len(grams)


# ------------------------------------------------------------------ #
# 표 출력 검증 — 신뢰도 점수가 없으니 출력 자체를 본다.
# ------------------------------------------------------------------ #
# 교체해도 안전한 실패 사유 — "읽다가 망가졌다" 는 증거가 있는 것만.
# ★ 왜 나누는가: KDoc 예측 900쪽을 재보니 검출 표 2,097개 중 실패 907개인데,
#   그중 568개는 8px 미만 극소 박스(오검출)라 page 로 회수할 대상이 아니고,
#   136개는 셀이 적을 뿐(2×2 장식표일 수 있다). 이걸 전부 교체하면 **정상적으로 읽힌
#   작은 표를 page 출력으로 덮어써 오히려 깎는다**. 그래서 기본은 strict 다.
HARD_FAILURES = ("truncated", "row_len_inconsistent", "otsl_parse_error",
                 "no_html_cells", "empty")
SOFT_FAILURES = ("too_few_cells", "no_grid", "no_rows")


def is_replaceable(reason: str, level: str = "strict") -> bool:
    """실패 사유가 교체 대상인지. level='all' 이면 약한 사유까지 포함한다."""
    base = reason.split("(")[0]
    if base in HARD_FAILURES:
        return True
    return level == "all" and base in SOFT_FAILURES


def validate_table(otsl: str) -> Tuple[bool, str]:
    """(정상인가, 이유). 이유는 진단 로그용이라 실패할 때만 의미가 있다."""
    if not otsl or not otsl.strip():
        return False, "empty"
    if NL not in otsl and otsl.count(FCEL) + otsl.count(CHED) < MIN_CELLS:
        return False, "no_grid"
    rows = [r for r in otsl.split(NL)]
    if rows and not rows[-1].strip():
        rows = rows[:-1]                      # 마지막 <nl> 뒤 빈 조각
    if not rows:
        return False, "no_rows"
    cells = [len(re.findall("|".join(re.escape(t) for t in
                                    (FCEL, CHED, ECEL, LCEL, UCEL, XCEL)), r)) for r in rows]
    total = sum(cells)
    if total < MIN_CELLS:
        return False, f"too_few_cells({total})"
    # 잘림 — 마지막 행이 앞 행들보다 뚜렷하게 짧으면 토큰 상한에서 끊긴 모양이다.
    if len(cells) >= 3:
        body = cells[:-1]
        mode = max(set(body), key=body.count)
        if mode and cells[-1] < mode * 0.5:
            return False, f"truncated(last={cells[-1]} vs {mode})"
        ok = sum(1 for c in body if c == mode)
        if ok / len(body) < MIN_ROW_CONSISTENCY:
            return False, "row_len_inconsistent"
    try:
        html = otsl_to_html(otsl)
    except Exception as e:                     # 파서가 못 읽으면 렌더도 못 한다
        return False, f"otsl_parse_error({type(e).__name__})"
    if "<td" not in html and "<th" not in html:
        return False, "no_html_cells"
    return True, ""


# ------------------------------------------------------------------ #
# page 출력 → 블록. 빈 줄로 끊고, HTML 표는 한 덩어리로 유지한다.
# ------------------------------------------------------------------ #
def md_blocks(page_md: str) -> List[str]:
    if not page_md:
        return []
    blocks, buf, in_table = [], [], False
    for line in page_md.splitlines():
        low = line.lower()
        if "<table" in low:
            in_table = True
        if in_table:
            buf.append(line)
            if "</table>" in low:
                blocks.append("\n".join(buf).strip()); buf, in_table = [], False
            continue
        if line.strip():
            buf.append(line)
        elif buf:
            blocks.append("\n".join(buf).strip()); buf = []
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]


def looks_like_table(block: str) -> bool:
    return bool(_TABLE_HINT.search(block))


# ------------------------------------------------------------------ #
# 레이아웃이 못 덮은 세로 밴드 — 좌표가 있으니 기하로 정해진다.
# ------------------------------------------------------------------ #
def uncovered_bands(elements: List[Dict], min_h: int = MIN_BAND_H,
                    top: int = 0, bottom: int = 1000) -> List[Tuple[int, int]]:
    spans = sorted((int(e["box"][1]), int(e["box"][3])) for e in elements
                   if e.get("box") and len(e["box"]) == 4)
    bands, cur = [], top
    for y0, y1 in spans:
        if y0 - cur >= min_h:
            bands.append((cur, y0))
        cur = max(cur, y1)
    if bottom - cur >= min_h:
        bands.append((cur, bottom))
    return bands


# ------------------------------------------------------------------ #
# 본체 — c2f 요소 목록을 골격으로 두고 page 결과를 보충한다.
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# hybrid-page — page 통읽기를 **골격**으로 두고 c2f 에서 표·머리말 정보만 가져온다.
#
# 왜 골격을 바꾸는가(같은 체크포인트 Stage 3 실측, KDoc 51쪽):
#     모드          표      긴 텍스트   머리말
#     c2f          27.8%    60.3%     89.8%
#     page 통읽기   15.7%    66.3%     85.7%
#   본문은 page 가 6.0%p 낫고 표는 c2f 가 12.1%p 낫다. hybrid-c2f 는 좋은 쪽(page 본문)을
#   보충재로만 써서 그 6%p 를 흘렸다 — 그래서 골격을 뒤집는다.
#
# ★ 머리말이 이 설계의 비용이다. KDoc 의 header_footer 는 "출력에 **없어야** 한다" 는
#   absent 테스트인데 page 통읽기는 쪽번호·기관명까지 그대로 전사한다(85.7%).
#   c2f 는 클래스로 버려서 89.8% 였다. 그래서 **c2f 가 읽은 Page-header/Page-footer
#   문자열을 page 출력에서 지운다** — 규칙으로 추측하지 않고 분류기 결과를 쓴다.
# ------------------------------------------------------------------ #
_PAGENO = re.compile(r"^[\s\-–—|]*(?:[0-9]{1,4}|[ivxlcIVXLC]{1,6}|-\s*\d+\s*-)[\s\-–—|]*$")


def strip_header_footer(page_md: str, elements: List[Dict], *,
                        overlap: float = 0.6) -> Tuple[str, int]:
    """page 출력에서 머리말·꼬리말 블록을 지운다. (결과, 지운 블록 수).

    ① c2f 가 Page-header/Page-footer 로 분류한 **문자열과 겹치는 블록**을 지운다.
    ② c2f 가 놓친 경우를 위해 첫·마지막 블록이 쪽번호꼴이면 지운다(보수적으로 양 끝만).
    """
    blocks = md_blocks(page_md)
    if not blocks:
        return page_md, 0
    hf = [norm_text(e.get("text", "")) for e in elements
          if e.get("label") in ("Page-header", "Page-footer") and e.get("text")]
    keep, dropped = [], 0
    for i, b in enumerate(blocks):
        nb = norm_text(b)
        # ★ 길이 조건이 없으면 **본문이 머리말로 오인돼 통째로 지워진다.**
        #   실측(2단 학회지 p02): 머리말 "이주여성의 뷰티관심도와 패션관심도가 패션관리행동에
        #   미치는 영향"(29자)의 3-gram 80%가 본문 "본 연구의 문제는 첫째, 뷰티관심도와
        #   패션관심도, 패션관리행동의 요인 간의…"(101자) 안에 들어 있어 문단 하나가 삭제됐다.
        #   머리말·꼬리말은 짧다 — 블록 길이가 머리말의 1.6배를 넘으면 그건 본문이다.
        if nb and any(len(h) >= 4 and len(nb) <= len(h) * 1.6
                      and (overlap_ratio(h, b) >= overlap or h in nb) for h in hf):
            dropped += 1
            continue
        if i in (0, len(blocks) - 1) and _PAGENO.match(b.strip()):
            dropped += 1
            continue
        keep.append(b)
    return "\n\n".join(keep), dropped


# body_fill 에서 **절대 첨부하면 안 되는** 라벨. KDoc header_footer 는 전부
# `type: absent`(출력에 없어야 통과)라, 이 두 라벨을 붙이면 96.97 → 68.94 로 무너진다.
BODY_SKIP = {"Table", "Page-header", "Page-footer", "Picture", "Formula"}


def _squash(t: str) -> str:
    """공백을 지운 비교용 문자열 — 줄바꿈·들여쓰기 차이로 중복을 놓치지 않게 한다."""
    return re.sub(r"\s+", "", t or "")


def merge_page_base(elements: List[Dict], page_md: str, *,
                    replace_level: str = "strict", page_w: int = 1400, page_h: int = 1980,
                    anchor_overlap: float = 0.55, drop_hf: bool = True,
                    place: str = "replace", allow_position_insert: bool = False,
                    unique_margin: float = 0.12, body_fill: bool = False) -> Tuple[str, Dict]:
    """page 골격 + c2f 표 이식. (Markdown, 진단)

    ★ **KDoc 채점은 위치를 보지 않는다**(README 실측):
        text_present  "출력 어디에든 나타나는가"
        tables        "어느 표에든 그 셀이 있고 이웃이 맞는가" — 표 안의 인접만 본다
        header_footer "출력에 없어야 한다"(absent)
      그래서 표를 제자리에 꽂으려고 평문을 교체하는 것은 **점수에는 이득이 없고
      본문을 훼손할 위험만 있다.** 기본은 안전한 쪽(교체 최소화 + 나머지는 뒤에 붙임)이다.
      제품 품질(사람이 읽는 Markdown)에는 위치가 중요하므로 anchor 모드를 남겨 둔다.

    place:
      "append"  — 아무것도 교체하지 않고 표를 **끝에 붙인다**. 본문 손실 0(ablation 기준선)
      "replace" — page 가 이미 표꼴로 낸 블록만 교체(강한 근거). 나머지는 끝에 붙인다
      "anchor"  — 위 + 셀 문자열 anchor 로 산문 블록까지 교체(유일하게 일치할 때만)
    allow_position_insert:
      상대 y 위치로 삽입하는 마지막 수단. 근거가 약해 **기본 off** 다 —
      잘못 꽂으면 읽기순서와 본문이 함께 망가지는데, 점수 이득은 0 이다.
    body_fill:
      ★ **page 골격이 못 덮은 본문 crop 을 끝에 붙인다**(V3.5 §4.5, 849쪽 실측).
      이 함수는 원래 표 crop 만 이식하고 `Text` `List-item` `Section-header`
      `Caption` `Footnote` `Title` crop 을 **한 글자도 쓰지 않았다.** page 통읽기가
      반복 루프로 무너진 페이지에서는 잘 읽어 둔 crop 까지 같이 버려진다
      (포항소식지 14: crop 51개가 1,684자를 읽었는데 최종 md 는 333자, 채점 0/127).

      849쪽 재조립 실측 — Long Text 62.93 → **74.74**, 3축 평균 67.44 → **71.29**
      (+3.85p, 페이지 부트스트랩 95% CI [+3.23, +4.52]). 표 점수는 42.42 로 불변이다.

      함정 둘.
        · `Page-header`/`Page-footer` crop 을 넣으면 HF 가 96.97 → **68.94** 로 붕괴한다.
          그래서 여기서는 그 두 라벨을 무조건 제외한다.
        · c2f 골격 단독으로 갈아타면 오히려 −5.42p 다. 이득의 원천은 **page ∪ crop
          합집합**이지 crop 쪽이 더 나아서가 아니다.

      대가: 뒤에 붙이므로 읽기순서가 흐트러진다(md 길이 ×1.09). KDoc 3축은 위치를
      채점하지 않아 점수 손해가 없을 뿐, **사람이 읽는 산출물 품질은 이 옵션 이전이 낫다.**
      제품 경로는 좌표 기반 병합으로 따로 고친다. 기본 off 인 이유다.
    """
    rep = {"tables_c2f": 0, "tables_used": 0, "by_pipe": 0, "by_anchor": 0,
           "by_position": 0, "hf_dropped": 0, "skipped_tiny": 0, "skipped_bad": 0,
           "body_seen": 0, "body_filled": 0, "body_chars": 0}
    md = page_md or ""
    if drop_hf:
        md, rep["hf_dropped"] = strip_header_footer(md, elements)
    blocks = md_blocks(md)

    tables = []
    for e in elements:
        if e.get("label") != "Table":
            continue
        rep["tables_c2f"] += 1
        b = e.get("box") or [0, 0, 0, 0]
        if (b[2] - b[0]) / 1000 * page_w < 8 or (b[3] - b[1]) / 1000 * page_h < 8:
            rep["skipped_tiny"] += 1                   # 8px 미만 = 오검출
            continue
        otsl = e.get("text", "")
        ok, why = validate_table(otsl)
        if not ok and not is_replaceable(why, replace_level):
            rep["skipped_bad"] += 1
            continue
        if not otsl.strip():
            rep["skipped_bad"] += 1
            continue
        tables.append((b, otsl, otsl_to_html(otsl)))
    tables.sort(key=lambda t: t[0][1])                 # 위→아래(읽기순서)

    used = [False] * len(blocks)
    pipe_idx = [i for i, b in enumerate(blocks) if looks_like_table(b)]
    out = list(blocks)
    inserts, appends = [], []
    rep["appended"] = 0
    rep["abstained"] = 0
    for box, otsl, html in tables:
        rep["tables_used"] += 1
        # ① page 가 이미 표꼴로 낸 블록 — 근거가 가장 강하다(둘 다 표다)
        cand = next((i for i in pipe_idx if not used[i]), None) if place != "append" else None
        if cand is not None:
            out[cand] = html
            used[cand] = True
            rep["by_pipe"] += 1
            continue
        # ② anchor — 산문으로 펼쳐진 블록. **유일하게** 일치할 때만 교체한다.
        #    2등과의 차이가 unique_margin 미만이면 후보가 여럿이라는 뜻이라 포기한다
        #    (리뷰 지적: 복수 후보에 억지로 꽂으면 본문·읽기순서까지 훼손된다).
        if place == "anchor":
            cells = [c for c in re.split(r"<[a-z]+>", otsl) if len(c.strip()) >= 2][:40]
            probe = " ".join(cells)
            scored = sorted(((overlap_ratio(b, probe), i) for i, b in enumerate(blocks)
                             if not used[i] and not looks_like_table(b)), reverse=True)
            if scored and scored[0][0] >= anchor_overlap and \
               (len(scored) == 1 or scored[0][0] - scored[1][0] >= unique_margin):
                out[scored[0][1]] = html
                used[scored[0][1]] = True
                rep["by_anchor"] += 1
                continue
            rep["abstained"] += 1
        # ③ 위치 삽입은 기본 off — 점수 이득 0, 훼손 위험만 있다.
        if allow_position_insert:
            pos = min(len(blocks), max(0, round(box[1] / 1000 * max(len(blocks), 1))))
            inserts.append((pos, html))
            rep["by_position"] += 1
        else:
            appends.append(html)                       # 끝에 붙인다(본문 손실 0)
            rep["appended"] += 1

    for pos, html in sorted(inserts, key=lambda x: -x[0]):
        out.insert(pos, html)
    out += appends

    # ── 표에 이미 들어 있는 낱개 조각을 지운다 ──
    # page 통읽기가 표를 격자로 못 보고 셀을 한 줄씩 흩뿌리는 경우가 있다
    # (실측 학회지 p04: ".902", "253", "자유도", "근사 카이제곱" … 9개가 표 뒤에
    # 낱개 블록으로 남았다). 같은 내용이 c2f 표 안에 이미 있으므로 **중복**이다.
    # 표 밖 본문까지 지우지 않도록 60자 이하 조각만, 표 셀에 그대로 들어 있을 때만.
    if tables:
        table_html = {h for _, _, h in tables}
        cell_txt = _squash(" ".join(re.sub(r"<[^>]+>", " ", h) for _, _, h in tables))
        cleaned, dropped = [], 0
        for b in out:
            sb = _squash(re.sub(r"<[^>]+>", " ", b))
            if b not in table_html and 0 < len(sb) <= 60 and sb in cell_txt:
                dropped += 1
                continue
            cleaned.append(b)
        out, rep["dup_cells_dropped"] = cleaned, dropped

    # ── body_fill: page 골격이 못 덮은 본문 crop 만 끝에 붙인다 ──
    if body_fill:
        skeleton = _squash(" ".join(b for b in out if b))
        for e in elements:
            lb = e.get("label")
            if lb in BODY_SKIP or not (e.get("text") or "").strip():
                continue
            rep["body_seen"] += 1
            raw = e["text"]
            if _squash(raw) in skeleton:          # 이미 골격에 있다 — 중복 첨부 금지
                continue
            out.append(raw)
            skeleton += _squash(raw)
            rep["body_filled"] += 1
            rep["body_chars"] += len(raw)

    return "\n\n".join(b for b in out if b and b.strip()), rep


def merge_hybrid(elements: List[Dict], page_md: str, *,
                 dup_overlap: float = DUP_OVERLAP,
                 min_band_h: int = MIN_BAND_H,
                 fill_bands: bool = True,
                 replace_level: str = "strict",
                 page_w: int = 1400, page_h: int = 1980) -> Tuple[List[Dict], Dict]:
    """(병합된 요소 목록, 진단). to_markdown 에 그대로 넘길 수 있는 형태로 돌려준다.

    요소에 붙는 필드:
      `hybrid`  — 'table_replaced' / 'band_filled' 중 하나(진단용)
      `text`    — 교체된 경우 page 쪽 문자열
      `label`   — 교체된 표는 'Table_html'(OTSL 아님을 조립기에 알린다)
    """
    els = [dict(e) for e in elements]
    blocks = md_blocks(page_md)
    used = [False] * len(blocks)
    c2f_all = " ".join(e.get("text", "") for e in els)
    rep = {"tables_total": 0, "tables_bad": 0, "tables_replaced": 0,
           "tables_skipped_tiny": 0, "tables_skipped_soft": 0,
           "bands": 0, "bands_filled": 0, "reasons": {}}

    # ── ① 실패한 표를 page 쪽 표 블록으로 교체 ──
    table_idx = [i for i, e in enumerate(els) if e.get("label") == "Table"]
    rep["tables_total"] = len(table_idx)
    page_tables = [i for i, b in enumerate(blocks) if looks_like_table(b)]
    for i in table_idx:
        ok, why = validate_table(els[i].get("text", ""))
        if ok:
            continue
        rep["tables_bad"] += 1
        rep["reasons"][why] = rep["reasons"].get(why, 0) + 1
        # ★ 극소 박스는 표가 아니다 — crop 자체를 건너뛴 오검출이라(원본에서 8px 미만)
        #   page 표를 끌어오면 있지도 않은 표를 만들어 넣는다. 실측 900쪽에서 실패 표의
        #   63%(568/907)가 이 경우였다.
        b = els[i].get("box") or [0, 0, 0, 0]
        if (b[2] - b[0]) / 1000 * page_w < 8 or (b[3] - b[1]) / 1000 * page_h < 8:
            rep["tables_skipped_tiny"] += 1
            continue
        if not is_replaceable(why, replace_level):
            rep["tables_skipped_soft"] += 1
            continue
        # 읽기순서상 이 표가 몇 번째 표인지 → page 쪽 같은 순번의 표 블록을 쓴다.
        rank = table_idx.index(i)
        cand = None
        if rank < len(page_tables) and not used[page_tables[rank]]:
            cand = page_tables[rank]
        else:                                   # 순번이 어긋나면 아직 안 쓴 표 블록 중 첫 번째
            for j in page_tables:
                if not used[j]:
                    cand = j
                    break
        if cand is None:
            continue
        els[i]["text"] = blocks[cand]
        els[i]["label"] = "Table_html"          # 이미 HTML/파이프 표 → 재변환 금지
        els[i]["hybrid"] = "table_replaced"
        used[cand] = True
        rep["tables_replaced"] += 1

    # ── ② 레이아웃이 못 덮은 밴드를 page 블록으로 보충 ──
    if fill_bands:
        bands = uncovered_bands(els, min_h=min_band_h)
        rep["bands"] = len(bands)
        for (by0, _by1) in bands:
            # 밴드 위치에 해당하는 삽입 지점 — 이 y 보다 아래에서 시작하는 첫 요소 앞.
            pos = next((k for k, e in enumerate(els)
                        if e.get("box") and e["box"][1] >= by0), len(els))
            for j, b in enumerate(blocks):
                if used[j] or looks_like_table(b):
                    continue
                if overlap_ratio(b, c2f_all) >= dup_overlap:
                    continue                     # c2f 가 이미 읽은 내용
                els.insert(pos, {"label": "Text", "text": b,
                                 "box": [0, by0, 1000, by0], "hybrid": "band_filled"})
                used[j] = True
                rep["bands_filled"] += 1
                break
    return els, rep


# ------------------------------------------------------------------ #
# 단(column)을 인식한 미검출 밴드 — 2단 조판에서 한쪽 단만 비는 경우를 잡는다.
#
# ★ 왜 필요한가: uncovered_bands() 는 페이지 전체를 **가로줄**로만 본다. 2단 문서에서
#   왼쪽 단이 차 있으면 그 y 구간은 "덮였다"고 판정하므로, **오른쪽 단만 통째로 비어도
#   보이지 않는다**(2단 학회지 p02: 우단 y131~488 두 문단이 이렇게 사라졌다).
#   단을 먼저 나눈 뒤 단별로 빈 구간을 찾으면 그 자리가 기하로 드러난다.
# ------------------------------------------------------------------ #
FULL_WIDTH_RATIO = 0.6      # 이 비율 이상 넓으면 전폭 요소(표·제목) — 모든 단을 덮는다
MIN_COL_BOXES    = 2        # 한 단으로 인정할 최소 박스 수
BAND_MIN_H       = 60       # 이 높이 미만의 빈틈은 무시(0~1000 기준)


def column_spans(elements: List[Dict], full_ratio: float = FULL_WIDTH_RATIO) -> List[Tuple[float, float]]:
    """박스 분포로 단 경계를 추정한다. 1단이면 [(0,1000)], 2단이면 [(0,mid),(mid,1000)]."""
    boxes = [e["box"] for e in elements if e.get("box") and len(e["box"]) == 4]
    narrow = [b for b in boxes if (b[2] - b[0]) < full_ratio * 1000]
    left  = [b for b in narrow if (b[0] + b[2]) / 2 < 500]
    right = [b for b in narrow if (b[0] + b[2]) / 2 >= 500]
    if len(left) < MIN_COL_BOXES or len(right) < MIN_COL_BOXES:
        return [(0.0, 1000.0)]
    lx1 = max(b[2] for b in left)
    rx0 = min(b[0] for b in right)
    mid = (lx1 + rx0) / 2 if rx0 > lx1 else 500.0
    return [(0.0, mid), (mid, 1000.0)]


def column_uncovered_bands(elements: List[Dict], *, min_h: int = BAND_MIN_H,
                           top: int = 60, bottom: int = 940,
                           full_ratio: float = FULL_WIDTH_RATIO) -> List[Tuple[float, float, float, float]]:
    """단별로 박스가 없는 구간을 찾아 [(x0, y0, x1, y1)] 로 돌려준다(0~1000 좌표).

    top/bottom 기본값은 머리말·꼬리말 영역을 뺀 본문 범위다 — 여백까지 재검출하면
    빈 crop 에 모델을 돌려 헛것을 만들 위험이 있다.
    """
    cols = column_spans(elements, full_ratio)
    out = []
    for cx0, cx1 in cols:
        spans = []
        for e in elements:
            b = e.get("box")
            if not b or len(b) != 4:
                continue
            wide = (b[2] - b[0]) >= full_ratio * 1000
            cx = (b[0] + b[2]) / 2
            if wide or (cx0 <= cx <= cx1):          # 전폭 요소는 모든 단을 덮는다
                spans.append((b[1], b[3]))
        cur = top
        for y0, y1 in sorted(spans):
            if y0 - cur >= min_h:
                out.append((cx0, cur, cx1, y0))
            cur = max(cur, y1)
        if bottom - cur >= min_h:
            out.append((cx0, cur, cx1, bottom))
    return out


def box_iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-6, ua)


def insert_by_reading_order(elements: List[Dict], new_els: List[Dict],
                            full_ratio: float = FULL_WIDTH_RATIO) -> List[Dict]:
    """새 박스를 읽기순서 제자리에 끼운다 — 같은 단에서 자기보다 아래인 첫 요소 앞."""
    cols = column_spans(elements, full_ratio)

    def col_of(b):
        cx = (b[0] + b[2]) / 2
        for i, (x0, x1) in enumerate(cols):
            if x0 <= cx <= x1:
                return i
        return len(cols) - 1

    out = list(elements)
    for ne in sorted(new_els, key=lambda e: (col_of(e["box"]), e["box"][1])):
        ci, y0 = col_of(ne["box"]), ne["box"][1]
        pos = len(out)
        for i, e in enumerate(out):
            b = e.get("box")
            if not b:
                continue
            if col_of(b) > ci or (col_of(b) == ci and b[1] > y0):
                pos = i
                break
        out.insert(pos, ne)
    return out
