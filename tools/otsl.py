"""OTSL(Optimized Table Structure Language) 변환기 — HTML 표 ↔ OTSL.

왜 OTSL 인가: MinerU2.5 가 표 인식 출력으로 채택한 직렬화로, HTML 대비 토큰 수를
절반 가까이 줄이고 구조 오류(태그 미폐합)를 문법적으로 차단한다. PierrotOCRVLM 의
`Table Recognition:` 태스크 정답 형식이다. 학습 데이터(sparsetables 등의 HTML GT)를
OTSL 로 변환해 쓰고, 추론 후에는 역변환으로 HTML 을 복원해 TEDS 평가·렌더 검증에 쓴다.

토큰 문법(OTSL 논문 규약):
    <fcel>텍스트   내용 있는 셀 시작(뒤에 셀 텍스트가 따라온다)
    <ched>텍스트   내용 있는 **헤더 셀**(HTML 의 <th> 또는 <thead> 안의 셀)
    <ecel>        빈 셀
    <lcel>        왼쪽 셀의 colspan 연장
    <ucel>        위쪽 셀의 rowspan 연장
    <xcel>        좌상단 셀의 rowspan+colspan 동시 연장(2D 병합 내부)
    <nl>          행 종료
격자를 행 우선으로 훑으며 병합 영역의 원점만 <fcel>/<ched>/<ecel> 이고 나머지는
l/u/x 로 채워진다 — span 숫자를 별도로 쓰지 않아도 병합이 복원된다.

★ <ched> 를 넣은 이유(2026-08-11): 표 벤치마크의 셀 테스트는 "이 셀의 위쪽/왼쪽
  **헤딩**이 무엇인가"를 묻는다. 헤더 표시가 하나도 없으면 채점기가 표 끝까지 걸어가
  엉뚱한 셀을 헤딩으로 집는다 — KDoc 표 실패의 65%가 이 항목이었다. PubTabNet 원천
  HTML 은 100% <thead> 를 갖고 있는데 변환 과정에서 우리가 버리고 있었다.
  빈 헤더 셀은 <ecel> 로 떨어뜨린다(헤더성 상실, 왕복은 일관 유지).

HTML 파싱은 stdlib html.parser 만 쓴다(빌더가 어디서든 돌게). 지원 범위는
표 데이터 GT 에 실제로 나오는 <table>/<tr>/<td>/<th>/<thead>/rowspan/colspan 이며,
셀 내부의 기타 태그는 텍스트만 남긴다. 단 <br>/<p>/<div>/<li> 는 **공백 한 칸**으로
바꾼다 — 이 태그를 그냥 버리면 여러 줄 셀이 "데이터제공량" 처럼 붙어버려, 완전일치를
요구하는 셀 테스트에서 전부 실패한다(KDoc 실측 손실 +9.7pt).
"""

from __future__ import annotations

from html import escape, unescape
from html.parser import HTMLParser
from typing import List, Optional, Tuple

FCEL, ECEL, LCEL, UCEL, XCEL, NL = "<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>"
CHED = "<ched>"                                   # 헤더 셀(= HTML <th> / <thead> 안의 셀)
_TOKENS = (FCEL, CHED, ECEL, LCEL, UCEL, XCEL, NL)

# 셀 안에서 **공백 한 칸**으로 치환할 태그. 여러 줄 셀이 붙는 것을 막는다.
_BREAK_TAGS = ("br", "p", "div", "li")


# ================================================================== #
# HTML → 셀 격자
# ================================================================== #

class _TableParser(HTMLParser):
    """<table> 하나를 [행][(rowspan, colspan, text, header)] 목록으로 파싱한다."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: List[List[Tuple[int, int, str, bool]]] = []
        self._row: Optional[List[Tuple[int, int, str, bool]]] = None
        self._cell: Optional[Tuple[int, int, bool]] = None
        self._text: List[str] = []
        self._depth = 0                      # 중첩 <table> 은 텍스트로 취급(지원 밖)
        self._thead = 0                      # <thead> 중첩 깊이 — 안쪽 셀은 전부 헤더

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
            return
        if self._depth != 1:
            return
        if tag == "thead":
            self._thead += 1
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            a = dict(attrs)

            def _span(key):
                try:
                    return max(1, int(a.get(key, 1) or 1))
                except (TypeError, ValueError):
                    return 1
            self._cell = (_span("rowspan"), _span("colspan"), tag == "th" or self._thead > 0)
            self._text = []
        elif tag in _BREAK_TAGS and self._cell is not None:
            self._text.append(" ")           # 줄바꿈은 공백 — 여러 줄 셀이 붙는 것을 막는다

    def handle_endtag(self, tag):
        if tag == "table":
            self._depth -= 1
            return
        if self._depth != 1:
            return
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            rs, cs, hd = self._cell
            self._row.append((rs, cs, " ".join("".join(self._text).split()), hd))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "thead":
            self._thead = max(0, self._thead - 1)
        elif tag in _BREAK_TAGS and self._cell is not None:
            self._text.append(" ")           # <p>…</p> 처럼 닫힘에서도 줄이 끊긴다

    def handle_data(self, data):
        if self._cell is not None:
            self._text.append(data)


# ------------------------------------------------------------------ #
# 행별 (rowspan, colspan, text, header) → 점유 격자.
# 격자 칸 값: ("origin", text, header) | ("l",) | ("u",) | ("x",)
# rowspan 이월을 추적하며 각 셀을 왼쪽부터 빈 칸에 배치한다(HTML 표 규칙).
# ------------------------------------------------------------------ #
def _rows_to_grid(rows):
    grid: List[List[Optional[tuple]]] = []

    def _ensure(r: int, c: int):
        while len(grid) <= r:
            grid.append([])
        row = grid[r]
        while len(row) <= c:
            row.append(None)

    for r, cells in enumerate(rows):
        _ensure(r, 0)
        c = 0
        for rs, cs, text, hd in cells:
            while c < len(grid[r]) and grid[r][c] is not None:
                c += 1                                        # 위 행 rowspan 이 점유한 칸 건너뜀
            _ensure(r, c + cs - 1)
            for dr in range(rs):
                _ensure(r + dr, c + cs - 1)
                for dc in range(cs):
                    if grid[r + dr][c + dc] is not None:
                        continue                              # ★ 선점 칸 보존 — 겹치는 span(비정형
                        #   HTML)에서 먼저 배치된 셀이 이긴다. 덮어쓰면 어느 사각형에도
                        #   속하지 않는 토큰이 생겨 OTSL 이 비일관해진다.
                    kind = ("origin" if (dr == 0 and dc == 0)
                            else "l" if dr == 0
                            else "u" if dc == 0
                            else "x")
                    grid[r + dr][c + dc] = (kind, text, hd) if kind == "origin" else (kind,)
            c += cs

    width = max((len(row) for row in grid), default=0)
    for row in grid:
        while len(row) < width:
            row.append(None)
    return grid


# ================================================================== #
# 공개 API
# ================================================================== #

# ------------------------------------------------------------------ #
# HTML 표 문자열 → OTSL 문자열. 표가 없거나 비어 있으면 None.
# ------------------------------------------------------------------ #
def html_to_otsl(html: str) -> Optional[str]:
    parser = _TableParser()
    parser.feed(html)
    if not parser.rows:
        return None
    grid = _rows_to_grid(parser.rows)

    parts: List[str] = []
    for row in grid:
        for cell in row:
            if cell is None:                                  # 미점유 칸 = 빈 셀
                parts.append(ECEL)
            elif cell[0] == "origin":
                # 빈 헤더 셀은 <ecel> 로 떨어진다(헤더성 상실 — 왕복 일관성 유지가 우선).
                parts.append(((CHED if cell[2] else FCEL) + cell[1]) if cell[1] else ECEL)
            else:
                parts.append({"l": LCEL, "u": UCEL, "x": XCEL}[cell[0]])
        parts.append(NL)
    return "".join(parts)


# ------------------------------------------------------------------ #
# OTSL 문자열 → HTML 표 문자열(<table>...</table>). 평가(TEDS)·렌더 검증용.
# l/u/x 토큰으로부터 각 origin 셀의 rowspan/colspan 을 복원한다.
# 문법 오류(허용 밖 병합 배치)는 관대하게 처리한다 — 모델 출력 평가가 목적이므로
# 예외 대신 가능한 구조로 복원한다.
# ------------------------------------------------------------------ #
def otsl_to_html(otsl: str) -> str:
    # 토큰화: 태그 기준으로 자르고, <fcel> 뒤 텍스트는 그 셀의 내용이 된다.
    cells_grid: List[List[tuple]] = [[]]
    i = 0
    while i < len(otsl):
        if otsl.startswith(NL, i):
            cells_grid.append([])
            i += len(NL)
            continue
        matched = None
        for tok in (FCEL, CHED, ECEL, LCEL, UCEL, XCEL):
            if otsl.startswith(tok, i):
                matched = tok
                break
        if matched is None:                                   # 잡음 문자 — 건너뜀
            i += 1
            continue
        i += len(matched)
        if matched in (FCEL, CHED):
            j = i
            while j < len(otsl) and not any(otsl.startswith(t, j) for t in _TOKENS):
                j += 1
            cells_grid[-1].append(("h" if matched == CHED else "f", unescape(otsl[i:j]).strip()))
            i = j
        else:
            cells_grid[-1].append({ECEL: ("e",), LCEL: ("l",), UCEL: ("u",), XCEL: ("x",)}[matched])
    while cells_grid and not cells_grid[-1]:
        cells_grid.pop()
    if not cells_grid:
        return "<table></table>"

    width = max(len(r) for r in cells_grid)
    for r in cells_grid:
        while len(r) < width:
            r.append(("e",))

    # origin 셀의 span 복원: 오른쪽 연속 l/x = colspan, 아래 연속 u/x = rowspan.
    html_rows: List[str] = []
    for ri, row in enumerate(cells_grid):
        tds: List[str] = []
        for ci, cell in enumerate(row):
            if cell[0] in ("l", "u", "x"):
                continue                                      # 병합 내부 — origin 이 담당
            cs = 1
            while ci + cs < width and cells_grid[ri][ci + cs][0] in ("l", "x"):
                cs += 1
            rs = 1
            while ri + rs < len(cells_grid) and cells_grid[ri + rs][ci][0] in ("u", "x"):
                rs += 1
            attrs = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
            text  = escape(cell[1]) if cell[0] in ("f", "h") else ""
            tag   = "th" if cell[0] == "h" else "td"          # 헤더 셀은 <th> 로 복원
            tds.append(f"<{tag}{attrs}>{text}</{tag}>")
        html_rows.append("<tr>" + "".join(tds) + "</tr>")
    return "<table>" + "".join(html_rows) + "</table>"
