#!/usr/bin/env python
"""문서 파싱 3패널 재생 뷰어 — Unlimited-OCR 데모와 같은 화면을 우리 출력으로 만든다.

    좌  원본 페이지 + 레이아웃 박스(읽는 중인 블록 강조)
    중  RAW OUTPUT — `<|det|>Text [x0, y0, x1, y1]<|/det|>내용` 인터리브 스트림
    우  재조판 결과(제목·본문·표는 실제 표 격자, 수식은 LaTeX)

★ 정직 표기 — 저쪽은 **1패스 인터리브** 모델이라 중앙 스트림이 진짜 한 줄기다.
  우리는 coarse-to-fine 2패스(레이아웃 1회 → crop 인식 N회)라, 중앙 스트림은
  **두 패스의 실제 출력을 읽기순서로 엮은 것**이다. 좌표도 내용도 모델이 낸 값
  그대로이고 재생 순서만 우리가 정한다. 화면 상단에도 그렇게 적는다.

입력은 `benchmark/run_pages.py --save-layout --save-trace` 가 남긴 `*.layout.json`
(요소 박스 + 블록별 raw 출력)과 그 페이지 이미지다.

사용 예:
    python tools/make_demo_viewer.py \\
        --pages "results/demo_paper/pages/*.png" \\
        --layout-dir results/demo_paper/md \\
        --out results/demo_paper/viewer.html --title "한국의상디자인학회지 15(2)"
"""

from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import os
import re
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.compare_modes import md_blocks, norm, render_block          # noqa: E402
from tools.otsl import otsl_to_html                                    # noqa: E402

# 클래스별 색 — 좌 박스와 우 배지가 같은 색을 쓴다(눈으로 연결되게).
CLASS_COLOR = {
    "Title":          "#ff5c5c",
    "Section-header": "#ff9d3c",
    "Text":           "#4da3ff",
    "List-item":      "#4dd6c1",
    "Table":          "#c07bff",
    "Formula":        "#ffd93d",
    "Caption":        "#8be36b",
    "Picture":        "#8d94a8",
    "Page-header":    "#5c6478",
    "Page-footer":    "#5c6478",
    # 레이아웃 지도에 없던 블록 — 페이지 통읽기(hybrid-page)에서 복구된 것.
    "Recovered":      "#3fb950",
}
DEFAULT_COLOR = "#8d94a8"


def img_b64(path: str, max_w: int) -> tuple[str, int, int]:
    """페이지 이미지를 JPEG base64 로. (data URI, 원본 폭, 원본 높이) 반환."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), w, h


def _strip(s: str) -> str:
    """비교 전용 정규화 — HTML 태그까지 지운다(요소 html 과 md 블록을 같은 지평에 둔다)."""
    return norm(re.sub(r"<[^>]+>", "", s or ""))


def _cover(a: str, b: str) -> float:
    """a 의 12자 조각이 b 안에 **실제로 들어 있는** 비율.

    ★ 조각끼리 집합으로 비교하면 안 된다 — 두 문자열의 시작 위치가 몇 글자만 어긋나도
      같은 문장인데 0점이 나온다(p00 영문 Abstract·p01 하단이 이 때문에 박스를 잃었다).
    """
    if not a or not b:
        return 0.0
    # ★ 짧은 글(소제목·캡션)은 조각을 만들 수 없다. 여기서 0.0 을 돌려주면
    #   "1. 뷰티관심도" 같은 절 제목이 **무조건 짝을 잃고 박스가 사라진다.**
    if len(a) < 12:
        if a in b:
            return 1.0
        # 짧은 글은 오타 한 글자에도 포함 검사가 깨진다(실측: "1. 뷰티관심도" → "1. 분티관실도").
        import difflib
        best = max((difflib.SequenceMatcher(None, a, b[i:i + len(a) + 2]).ratio()
                    for i in range(0, max(1, len(b) - len(a)), 2)), default=0.0)
        return best
    g = [a[i:i + 12] for i in range(0, len(a) - 12, 3)] or [a[:12]]
    return sum(1 for c in g if c in b) / len(g)


def match_to_elements(blocks: list, els: list) -> list:
    """최종 마크다운 블록 ↔ 레이아웃 요소 짝짓기.

    짝이 있으면 그 요소의 박스를 그대로 쓰고(=지도가 찾아낸 블록), 정말 짝이 없는 것만
    Recovered = 지도에 없다가 페이지 통읽기로 살아난 블록이다.
    """
    THRESH = 0.30
    # ★ 머리말·꼬리말은 최종 마크다운에서 빠지므로 짝이 될 수 없다. 후보에 두면
    #   쪽번호("125")가 본문에 들어 있다는 이유로 긴 본문과 짝이 되어 **본문이
    #   Page-header 로 표시된다**(실측 p02). 아예 제외한다.
    pool = [(i, e, _strip(e.get("raw", "")), _strip(e.get("html", "")))
            for i, e in enumerate(els)
            if e.get("label") not in ("Page-header", "Page-footer")]
    used, out = set(), []
    for blk in blocks:
        nb = _strip(blk)
        best, score = None, 0.0
        for i, e, raw, htm in pool:
            if i in used:
                continue
            # 길이가 크게 다른 둘은 짝이 아니다 — 짧은 글의 포함 검사가 긴 본문에
            #   우연히 걸리는 것을 막는다.
            ref = raw or htm
            if ref and nb and max(len(ref), len(nb)) > min(len(ref), len(nb)) * 3 + 12:
                continue
            ov = max(_cover(raw, nb), _cover(nb, raw), _cover(htm, nb), _cover(nb, htm))
            if ov > score:
                best, score = (i, e), ov
        # 짧은 글끼리는 글자 유사도가 쉽게 올라간다("III. 연구방법" ↔ "제주리라해도").
        # 그래서 짧은 짝에는 더 높은 문턱을 요구한다 — 틀린 박스를 붙이느니 비워 둔다.
        need = THRESH
        if best:
            bl = min(len(_strip(best[1].get("raw", ""))), len(_strip(blk)))
            if bl < 12:
                need = 0.62
        if best and score >= need:
            i, e = best
            used.add(i)
            out.append({"label": e["label"], "box": e["box"], "_idx": i,
                        "raw": e.get("raw", "") or blk, "html": render_block(blk)})
        else:
            out.append({"label": "Recovered", "box": None, "_idx": None,
                        "raw": blk, "html": render_block(blk)})

    # ── 후처리: 글자로는 못 찾았지만 **순서로는 자리가 정해지는** 블록을 메운다.
    #   c2f 가 소제목을 헛읽으면("3. 패션관리행동" → "제주리라해도") 글자가 전혀 달라
    #   짝을 못 찾는다. 그런데 앞뒤 블록이 요소 i, j 에 붙어 있다면 그 사이에 남은
    #   요소가 곧 이 블록의 자리다. 개수가 정확히 맞을 때만 배정한다(추측 최소화).
    idxs = [o["_idx"] for o in out]
    k = 0
    while k < len(out):
        if idxs[k] is not None:
            k += 1
            continue
        j = k
        while j < len(out) and idxs[j] is None:
            j += 1
        prev = idxs[k - 1] if k > 0 else -1
        nxt = idxs[j] if j < len(out) else len(els)
        if prev is not None and nxt is not None:
            gap = [i for i in range(prev + 1, nxt) if i not in used]
            if len(gap) == j - k:
                for off, i in enumerate(gap):
                    e = els[i]
                    used.add(i)
                    out[k + off].update({"label": e["label"], "box": e["box"], "_idx": i})
                    idxs[k + off] = i
        k = j
    for o in out:
        o.pop("_idx", None)
    return out


def build_page(png: str, layout_json: str, max_w: int, md_path: str = "") -> dict | None:
    """페이지 하나 → {이미지, 요소 목록}. 요소에는 박스·클래스·raw·렌더HTML 이 담긴다."""
    if not os.path.exists(layout_json):
        return None
    diag = json.load(open(layout_json, encoding="utf-8"))
    els = diag.get("elements") or []
    trace = {t["i"]: t for t in diag.get("trace") or []}
    uri, w, h = img_b64(png, max_w)

    out = []
    for i, el in enumerate(els):
        t = trace.get(i, {})
        raw = t.get("raw", "")
        label = el["label"]
        if label == "Table":
            rendered = t.get("html") or (otsl_to_html(raw) if raw else "")
        elif label == "Formula":
            rendered = f'<div class="formula">$$ {html.escape(raw)} $$</div>'
        elif label == "Picture":
            rendered = '<div class="pic">🖼 (그림 — 인식 생략)</div>'
        elif label == "Title":
            rendered = f'<h1>{html.escape(raw)}</h1>'
        elif label == "Section-header":
            rendered = f'<h2>{html.escape(raw)}</h2>'
        elif label == "List-item":
            rendered = f'<p>• {html.escape(raw)}</p>'
        elif label == "Caption":
            rendered = f'<p><em>{html.escape(raw)}</em></p>'
        else:
            rendered = f'<p>{html.escape(raw)}</p>'
        out.append({"i": i, "label": label, "box": el["box"], "raw": raw,
                    "html": rendered})
    if md_path and os.path.exists(md_path):
        out = match_to_elements(md_blocks(md_path), out)
    for n, el in enumerate(out):
        el["i"] = n
    return {"name": os.path.basename(png), "img": uri, "w": w, "h": h, "elements": out}


HTML_TMPL = """<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg:#0d1117; --panel:#131923; --line:#243044; --fg:#e6edf3; --dim:#8b98ac;
    --accent:#eb6834;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:"Noto Sans KR",-apple-system,"Segoe UI",sans-serif; }
  header { display:flex; align-items:center; gap:14px; padding:10px 16px;
           border-bottom:1px solid var(--line); background:var(--panel); flex-wrap:wrap; }
  .logo { font-weight:800; font-size:20px; letter-spacing:-.5px; }
  .logo em { color:var(--accent); font-style:normal; }
  .sub { color:var(--dim); font-size:13px; }
  button, select { background:#1b2432; color:var(--fg); border:1px solid var(--line);
                   border-radius:6px; padding:6px 12px; font-size:14px; cursor:pointer; }
  button.go { background:var(--accent); border-color:var(--accent); color:#141414;
              font-weight:700; }
  .prog { flex:1; min-width:160px; height:6px; background:#1b2432; border-radius:3px;
          overflow:hidden; }
  .prog > i { display:block; height:100%; width:0; background:var(--accent); }
  main { display:grid; grid-template-columns:1fr 1fr 1fr; gap:1px; background:var(--line);
         height:calc(100vh - 108px); }
  section { background:var(--panel); overflow:auto; padding:12px; }
  h2.pt { position:sticky; top:-12px; margin:-12px -12px 10px; padding:8px 12px;
          font-size:12px; letter-spacing:2px; color:var(--accent); background:var(--panel);
          border-bottom:1px solid var(--line); text-transform:uppercase; z-index:5; }
  .pagewrap { position:relative; margin-bottom:14px; }
  .pagewrap img { width:100%; display:block; border-radius:4px; }
  .pagewrap .lbl { position:absolute; font-size:10px; padding:0 3px; border-radius:2px;
                   color:#0d1117; font-weight:700; transform:translateY(-100%); white-space:nowrap; }
  .bx { position:absolute; border:2px solid; border-radius:2px; opacity:0; transition:opacity .2s; }
  .bx.on { opacity:1; }
  .bx.cur { box-shadow:0 0 0 3px rgba(235,104,52,.55); background:rgba(235,104,52,.10); }
  #raw { font-family:"JetBrains Mono",Menlo,Consolas,monospace; font-size:13.5px;
         line-height:1.65; white-space:pre-wrap; word-break:break-all; color:#c9d6e5; }
  #raw .det { color:var(--accent); }
  #doc { font-size:15.5px; line-height:1.75; }
  #doc .blk { border:1px solid var(--line); border-radius:6px; padding:10px 12px;
              margin-bottom:10px; position:relative; }
  #doc .blk.cur { border-color:var(--accent); }
  #doc .badge { position:absolute; top:-9px; left:8px; font-size:10px; font-weight:700;
                padding:1px 6px; border-radius:3px; color:#0d1117; }
  #doc p { margin:0; }
  #doc h1 { font-size:22px; margin:0; }
  #doc h2 { font-size:18px; margin:0; }
  #doc table { border-collapse:collapse; width:100%; font-size:12.5px; }
  #doc td, #doc th { border:1px solid var(--line); padding:3px 6px; }
  #doc th { background:#1b2432; }
  #doc .formula { font-family:Menlo,monospace; color:#ffd93d; font-size:13px;
                  word-break:break-all; }
  #doc .pic { color:var(--dim); }
  .note { padding:6px 16px; font-size:12px; color:var(--dim); background:var(--panel);
          border-top:1px solid var(--line); }
</style>

<header>
  <div class="logo">Pierrot<em>-OCR</em></div>
  <div class="sub">__SUBTITLE__</div>
  <select id="speed">
    <option value="3">1x</option>
    <option value="8" selected>3x</option>
    <option value="25">10x</option>
    <option value="99999">즉시</option>
  </select>
  <button class="go" id="play">▶ START</button>
  <button id="reset">↺ 처음부터</button>
  <div class="prog"><i id="bar"></i></div>
  <div class="sub" id="stat">0 / 0 블록</div>
</header>

<main>
  <section id="pane-img"><h2 class="pt">Input · layout</h2><div id="pages"></div></section>
  <section id="pane-raw"><h2 class="pt">Raw output</h2><div id="raw"></div></section>
  <section id="pane-doc"><h2 class="pt">Parsed document</h2><div id="doc"></div></section>
</main>
<div class="note">__NOTE__</div>

<script>
const DATA = __DATA__;
const COLOR = __COLOR__;
const col = l => COLOR[l] || "__DEFCOLOR__";

// ── 좌 패널: 페이지 이미지와 박스 오버레이를 미리 깔아 둔다(박스는 숨김 상태) ──
const pagesEl = document.getElementById("pages");
const queue = [], labels = [];                                  // 재생 큐: 모든 페이지의 블록을 읽기순서로
DATA.forEach((pg, pi) => {
  const wrap = document.createElement("div");
  wrap.className = "pagewrap";
  wrap.innerHTML = `<img src="${pg.img}">`;
  pg.elements.forEach((el, ei) => {
    if (!el.box) { queue.push({pi, ei, el, box: null}); return; }   // 지도에 없던 블록
    const [x0, y0, x1, y1] = el.box;             // 0~999 정규 좌표 → %
    const b = document.createElement("div");
    b.className = "bx";
    b.style.cssText = `left:${x0/10}%;top:${y0/10}%;width:${(x1-x0)/10}%;` +
                      `height:${(y1-y0)/10}%;border-color:${col(el.label)}`;
    const lb = document.createElement("div");
    lb.className = "lbl";
    lb.textContent = el.label;
    lb.style.cssText = `left:${x0/10}%;top:${y0/10}%;background:${col(el.label)};opacity:0`;
    b.dataset.li = String(labels.length);
    wrap.appendChild(b); wrap.appendChild(lb);
    labels.push(lb);
    queue.push({pi, ei, el, box:b});
  });
  pagesEl.appendChild(wrap);
});

const rawEl = document.getElementById("raw"), docEl = document.getElementById("doc");
const bar = document.getElementById("bar"), stat = document.getElementById("stat");
let qi = 0, ci = 0, timer = null, blk = null, rawSpan = null;

function startBlock(item) {
  if (item.box) {
    item.box.classList.add("on", "cur");
    labels[+item.box.dataset.li].style.opacity = 1;
    item.box.scrollIntoView({block:"center", behavior:"smooth"});
  }
  // 중앙: det 헤더를 먼저 찍는다 — 모델이 낸 박스 좌표 그대로.
  const head = document.createElement("span");
  head.className = "det";
  head.textContent = item.el.box
    ? `<|det|>${item.el.label} [${item.el.box.join(", ")}]<|/det|>`
    : `<|page|>레이아웃 미검출 → 페이지 통읽기에서 복구<|/page|>`;
  rawEl.appendChild(head);
  rawSpan = document.createElement("span");
  rawEl.appendChild(rawSpan);
  // 우: 빈 카드부터 만들고 채운다.
  blk = document.createElement("div");
  blk.className = "blk cur";
  blk.innerHTML = `<span class="badge" style="background:${col(item.el.label)}">` +
                  `${item.el.label}</span><div class="body"></div>`;
  docEl.appendChild(blk);
}

function finishBlock(item) {
  if (item.box) item.box.classList.remove("cur");
  blk.classList.remove("cur");
  blk.querySelector(".body").innerHTML = item.el.html;   // 완성 시 실제 렌더로 교체
  rawEl.appendChild(document.createTextNode("\\n"));
  docEl.parentElement.scrollTop = docEl.parentElement.scrollHeight;
}

function tick() {
  const step = +document.getElementById("speed").value;
  if (qi >= queue.length) { stop(); return; }
  const item = queue[qi];
  if (ci === 0) startBlock(item);
  const raw = item.el.raw || "";
  const next = Math.min(raw.length, ci + step);
  rawSpan.textContent = raw.slice(0, next);
  // 표·수식은 타이핑 중 raw 를 그대로 보여 주고, 끝나면 렌더 결과로 바꾼다.
  blk.querySelector(".body").textContent = raw.slice(0, next);
  ci = next;
  rawEl.parentElement.scrollTop = rawEl.parentElement.scrollHeight;
  if (ci >= raw.length) {
    finishBlock(item);
    qi++; ci = 0;
    bar.style.width = (qi / queue.length * 100) + "%";
    stat.textContent = `${qi} / ${queue.length} 블록`;
  }
}

function play() { if (!timer) { timer = setInterval(tick, 16);
                               document.getElementById("play").textContent = "❚❚ STOP"; } }
function stop() { clearInterval(timer); timer = null;
                  document.getElementById("play").textContent = "▶ START"; }
document.getElementById("play").onclick = () => timer ? stop() : play();
document.getElementById("reset").onclick = () => {
  stop(); qi = 0; ci = 0; rawEl.textContent = ""; docEl.textContent = "";
  document.querySelectorAll(".bx").forEach(b => b.classList.remove("on", "cur"));
  labels.forEach(l => l.style.opacity = 0);
  bar.style.width = "0"; stat.textContent = `0 / ${queue.length} 블록`;
};
stat.textContent = `0 / ${queue.length} 블록`;
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="문서 파싱 3패널 재생 뷰어")
    ap.add_argument("--pages", required=True, help="페이지 이미지 glob")
    ap.add_argument("--layout-dir", required=True, help="run_pages --save-layout 출력 디렉토리")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="문서 파싱 데모")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--max-width", type=int, default=1100, help="이미지 인라인 폭(px)")
    ap.add_argument("--md-dir", default="",
                    help="최종 마크다운 디렉토리(hybrid-page 등). 주면 우패널을 이 결과로 만든다")
    a = ap.parse_args()

    pages = []
    for png in sorted(glob.glob(a.pages)):
        lj = os.path.join(a.layout_dir,
                          os.path.splitext(os.path.basename(png))[0] + ".layout.json")
        md = os.path.join(a.md_dir, os.path.splitext(os.path.basename(png))[0] + ".md") if a.md_dir else ""
        pg = build_page(png, lj, a.max_width, md)
        if pg:
            pages.append(pg)
        else:
            print(f"[viewer] 건너뜀(레이아웃 없음): {os.path.basename(png)}")
    if not pages:
        raise SystemExit("[viewer] 재생할 페이지가 없다 — run_pages 를 --save-layout --save-trace 로 돌렸는지 확인")

    n_el = sum(len(p["elements"]) for p in pages)
    note = ("좌·중·우 모두 PierrotOCRVLM 이 실제로 낸 출력이다. 좌표는 레이아웃 패스, "
            "내용은 crop 인식 패스의 결과이며 재생 순서는 모델이 낸 읽기순서를 따른다. "
            "우리 파이프라인은 coarse-to-fine 2패스라 중앙 스트림은 한 줄기 생성이 아니라 "
            "블록별 출력을 읽기순서로 이어 붙인 것이다.")
    doc = (HTML_TMPL
           .replace("__TITLE__", html.escape(a.title))
           .replace("__SUBTITLE__", html.escape(a.subtitle or f"{len(pages)}쪽 · {n_el}블록"))
           .replace("__NOTE__", note)
           .replace("__DATA__", json.dumps(pages, ensure_ascii=False).replace("</", "<\\/"))
           .replace("__COLOR__", json.dumps(CLASS_COLOR))
           .replace("__DEFCOLOR__", DEFAULT_COLOR))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[viewer] {len(pages)}쪽 {n_el}블록 → {a.out} "
          f"({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
