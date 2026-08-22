"""문서 파싱 평가 지표 (외부 의존 없이 stdlib 만 사용).

세 논문(MinerU2.5·PaddleOCR-VL·GLM-OCR)이 공통으로 쓰는 지표를 우리 태스크에 맞춰 구현한다:

  텍스트/페이지 : **NED**(normalized edit distance) — 낮을수록 좋다. 논문들의
                  "Text edit" 열과 같은 정의(레벤슈타인 / max(len)).
  수식          : NED + exact match. (CDM 은 렌더 기반이라 별도 인프라가 필요해
                  1차 평가에서는 제외 — M3 후반 과제.)
  표            : **TEDS / TEDS-Struct**(tree edit distance similarity, 높을수록 좋다).
                  OTSL → HTML 로 복원한 뒤 표를 트리로 보고 편집거리를 잰다.
                  TEDS-Struct 는 셀 내용을 무시하고 구조만 본다.
  레이아웃      : IoU 매칭 기반 **precision/recall/F1**(클래스 일치 요구) +
                  **읽기순서 상관**(매칭된 요소들의 Kendall tau).

레벤슈타인은 O(nm) 이라 긴 페이지(수천 자)에서 비싸다 — 행 2개만 유지하는
구현으로 메모리를 O(n) 으로 낮추고, 길이 상한을 넘으면 잘라서 잰다(평가 일관성을
위해 상한을 인자로 노출).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple


# ================================================================== #
# 편집거리 계열
# ================================================================== #

# ------------------------------------------------------------------ #
# 레벤슈타인 거리(행 2개만 유지). 시퀀스는 문자열/리스트 모두 가능.
# ------------------------------------------------------------------ #
def levenshtein(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # 삭제
                           cur[j - 1] + 1,        # 삽입
                           prev[j - 1] + (ca != cb)))  # 치환
        prev = cur
    return prev[-1]


# ------------------------------------------------------------------ #
# 정규화 편집거리(0=완전일치, 1=완전불일치). 논문들의 "edit distance" 열.
# 공백 정규화는 선택(문서 파싱 관례상 연속 공백은 하나로 본다).
# ------------------------------------------------------------------ #
def normalized_edit_distance(pred: str, gold: str, normalize_space: bool = True,
                             max_len: int = 4000) -> float:
    if normalize_space:
        pred = " ".join(pred.split())
        gold = " ".join(gold.split())
    pred, gold = pred[:max_len], gold[:max_len]
    if not pred and not gold:
        return 0.0
    return levenshtein(pred, gold) / max(len(pred), len(gold), 1)


# ================================================================== #
# 표: TEDS (tree edit distance similarity)
# ================================================================== #

_CELL_RE = re.compile(r"<t[dh]([^>]*)>(.*?)</t[dh]>", re.S)
_ROW_RE  = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_SPAN_RE = re.compile(r'(rowspan|colspan)\s*=\s*"?(\d+)"?')
_TAG_RE  = re.compile(r"<[^>]+>")


# ------------------------------------------------------------------ #
# HTML 표를 (행, 셀) 트리로 편다. 셀 = (rowspan, colspan, 텍스트).
# ------------------------------------------------------------------ #
def _parse_table(html: str) -> List[List[Tuple[int, int, str]]]:
    rows = []
    for row_html in _ROW_RE.findall(html):
        cells = []
        for attrs, inner in _CELL_RE.findall(row_html):
            spans = dict(_SPAN_RE.findall(attrs))
            rs = int(spans.get("rowspan", 1))
            cs = int(spans.get("colspan", 1))
            text = " ".join(_TAG_RE.sub("", inner).split())
            cells.append((rs, cs, text))
        rows.append(cells)
    return rows


# ------------------------------------------------------------------ #
# TEDS: 표를 "행 리스트 → 셀 리스트" 2단 트리로 보고, 행 단위 정렬 후
# 셀 편집거리를 합산해 유사도를 만든다(1 = 완전 일치).
#
# 원 논문의 TEDS 는 APTED 로 일반 트리 편집거리를 풀지만, 표는 깊이 2의
# 규칙적 트리라 행 정렬 + 셀 시퀀스 편집거리로 거의 같은 값을 얻는다
# (외부 의존 없이 돌리기 위한 실용적 근사 — 절대값 비교보다 **모델 간 상대
# 비교**에 쓰는 것이 목적이다).
#
# structure_only=True 면 셀 텍스트를 무시하고 (rowspan, colspan) 만 본다(TEDS-Struct).
# ------------------------------------------------------------------ #
def teds(pred_html: str, gold_html: str, structure_only: bool = False) -> float:
    pred_rows = _parse_table(pred_html)
    gold_rows = _parse_table(gold_html)
    if not gold_rows:
        return 1.0 if not pred_rows else 0.0
    if not pred_rows:
        return 0.0

    def key(cell):
        return (cell[0], cell[1]) if structure_only else (cell[0], cell[1], cell[2])

    # 행 시퀀스를 "셀 키 튜플"로 요약해 행 단위 정렬(삽입/삭제/치환) 비용을 구한다.
    pred_keys = [tuple(key(c) for c in r) for r in pred_rows]
    gold_keys = [tuple(key(c) for c in r) for r in gold_rows]

    # 행 단위 DP: 치환 비용 = 그 행의 셀 편집거리 / max(셀수)
    n, m = len(pred_keys), len(gold_keys)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + max(len(pred_keys[i - 1]), 1)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + max(len(gold_keys[j - 1]), 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = levenshtein(pred_keys[i - 1], gold_keys[j - 1])
            dp[i][j] = min(dp[i - 1][j] + max(len(pred_keys[i - 1]), 1),
                           dp[i][j - 1] + max(len(gold_keys[j - 1]), 1),
                           dp[i - 1][j - 1] + sub)
    total = max(sum(max(len(r), 1) for r in pred_keys),
                sum(max(len(r), 1) for r in gold_keys), 1)
    return max(0.0, 1.0 - dp[n][m] / total)


# ================================================================== #
# 레이아웃: IoU 매칭 + 읽기순서
# ================================================================== #

def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------ #
# 예측/정답 요소 리스트(각 {label, box}) 를 IoU 임계값으로 그리디 매칭해
# precision/recall/F1 과 읽기순서 상관(Kendall tau)을 낸다.
#   - 클래스가 다르면 매칭하지 않는다(검출+분류 동시 평가).
#   - 읽기순서: 매칭된 쌍의 (예측 순위, 정답 순위) 로 tau 계산(1=완전 일치).
# ------------------------------------------------------------------ #
def layout_scores(pred: List[Dict], gold: List[Dict], iou_thr: float = 0.5) -> Dict[str, float]:
    used = set()
    matches: List[Tuple[int, int]] = []
    for pi, p in enumerate(pred):
        best, best_iou = -1, iou_thr
        for gi, g in enumerate(gold):
            if gi in used or g["label"] != p["label"]:
                continue
            v = _iou(p["box"], g["box"])
            if v >= best_iou:
                best, best_iou = gi, v
        if best >= 0:
            used.add(best)
            matches.append((pi, best))

    tp = len(matches)
    precision = tp / len(pred) if pred else 0.0
    recall    = tp / len(gold) if gold else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Kendall tau (매칭 쌍이 2개 미만이면 정의 불가 → None 대신 1.0/0.0 회피 위해 -1 표기)
    tau = -1.0
    if len(matches) >= 2:
        conc = disc = 0
        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                a = matches[i][0] - matches[j][0]
                b = matches[i][1] - matches[j][1]
                if a * b > 0:
                    conc += 1
                elif a * b < 0:
                    disc += 1
        total = conc + disc
        tau = (conc - disc) / total if total else 1.0

    return {"precision": precision, "recall": recall, "f1": f1,
            "order_tau": tau, "n_pred": len(pred), "n_gold": len(gold)}


# ------------------------------------------------------------------ #
# detection-as-text 문자열 → [{label, box(0~999)}] 파싱.
# 형식: "{class}: x0,y0,x1,y1 ; ..." (모델 출력/정답 공통).
# ------------------------------------------------------------------ #
_DET_RE = re.compile(r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)")


def parse_layout(text: str) -> List[Dict]:
    out = []
    for seg in text.split(";"):
        m = _DET_RE.search(seg)
        if not m:
            continue
        box = [float(v) for v in m.groups()]
        label = seg[:m.start()].strip().rstrip(":").strip()
        out.append({"label": label, "box": box})
    return out

# ------------------------------------------------------------------ #
# LaTeX 표준형 — 방언 차이를 걷어내고 "수식이 같은가"만 남긴다.
#
# 왜 필요한가: 같은 수식을 UniMER 는 `{ 2 } & { 3 }`, OmniDocBench 는 `2 & 3` 으로
# 쓴다. 이런 차이로 exact match 가 0.048 까지 떨어져 **인식 실력이 아니라 표기
# 규약을 재게 된다**. 아래 규칙은 전부 "이미지에서 알 수 없는 것"만 지운다 —
# 정렬 지정자(ccc/lll)는 보이지 않고, 단일 토큰을 감싼 중괄호는 의미가 없으며,
# 식 번호는 수식 자체가 아니다. 예측·정답 **양쪽에 똑같이** 적용한다.
# ------------------------------------------------------------------ #
_TEX_ALIGN = re.compile(r"(\\begin\{(?:array|tabular)\})\s*\{[lcr|@{}\s.]*\}")
_TEX_TAG   = re.compile(r"^\s*\(\s*\d+[a-z]?\s*\)\s*(?:\\quad|\\qquad|~)?\s*")
_TEX_BRACE = re.compile(r"\{([^{}]*)\}")


def latex_canonical(s: str) -> str:
    """LaTeX 를 비교 가능한 표준형으로. 예측·정답 모두에 적용할 것."""
    s = s.strip()
    for w in ("$$", "$"):                                  # 수식 래퍼
        if s.startswith(w) and s.endswith(w) and len(s) > 2 * len(w):
            s = s[len(w):-len(w)].strip()
            break
    s = _TEX_TAG.sub("", s)                                # 앞머리 식 번호 "(9) \quad"
    s = s.replace("\\lbrack", "[").replace("\\rbrack", "]")
    s = s.replace("\\left", "").replace("\\right", "")
    s = _TEX_ALIGN.sub(r"\1", s)                          # 정렬 지정자 {ccc} 제거
    for _ in range(8):                                     # 중첩 중괄호 벗기기
        t = _TEX_BRACE.sub(r"\1", s)
        if t == s:
            break
        s = t
    return re.sub(r"\s+", "", s)
