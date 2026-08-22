#!/usr/bin/env python
"""같은 페이지를 두 모드로 파싱한 결과를 나란히 놓고 비교하는 이미지.

왼쪽에 원본, 가운데와 오른쪽에 두 마크다운 결과를 그대로 렌더한다. 한쪽에만 있는
문단은 초록 테두리로 표시해 "무엇이 살아났고 무엇이 사라졌는지"가 바로 보이게 한다.

playwright 가 있는 파이썬으로 실행한다:

    <pwenv>/bin/python tools/compare_modes.py --image results/demo_paper/pages/p02.png \\
        --md-a results/demo_paper/md/p02.md        --label-a "coarse-to-fine" \\
        --md-b results/demo_paper/md_hybridpage/p02.md --label-b "hybrid-page" \\
        --out results/demo_paper/compare_p02.png
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import re

from PIL import Image

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    """비교용 정규화 — 공백과 마크다운 장식을 지운다."""
    return _WS.sub("", re.sub(r"[#*`>\-|]", "", s)).lower()


def md_blocks(path: str) -> list[str]:
    """빈 줄로 나뉜 블록 목록. 표(<table>)는 한 덩어리로 유지된다."""
    return [b.strip() for b in open(path, encoding="utf-8").read().split("\n\n") if b.strip()]


def render_block(b: str) -> str:
    """아주 작은 마크다운 렌더 — 우리 출력에 쓰이는 문법만 다룬다."""
    if b.lstrip().startswith("<table"):
        return b
    if b.startswith("## "):
        return f"<h2>{html.escape(b[3:])}</h2>"
    if b.startswith("# "):
        return f"<h1>{html.escape(b[2:])}</h1>"
    if b.startswith("$$"):
        return f'<pre class="formula">{html.escape(b)}</pre>'
    if b.startswith("*") and b.endswith("*"):
        return f"<p><em>{html.escape(b.strip('*'))}</em></p>"
    return f"<p>{html.escape(b)}</p>"


def img_uri(path: str, max_w: int) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


TMPL = """<meta charset="utf-8">
<style>
  :root { --bg:#0d1117; --panel:#131923; --line:#2a3549; --fg:#e8eef6; --dim:#93a1b5;
          --accent:#eb6834; --new:#3fb950; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); width:__W__px;
         font-family:"Noto Sans KR",-apple-system,sans-serif; }
  .hd { padding:14px 20px; border-bottom:1px solid var(--line); background:var(--panel);
        display:flex; gap:14px; align-items:baseline; }
  .hd .t { font-weight:800; font-size:22px; } .hd .t em { color:var(--accent); font-style:normal; }
  .hd .s { color:var(--dim); font-size:15px; }
  .cols { display:flex; }
  .col { padding:16px 18px; width:__CW__px; border-right:1px solid var(--line); }
  .cap { font-size:14px; letter-spacing:1px; color:var(--accent); margin:0 0 12px;
         text-transform:uppercase; }
  .cap b { color:var(--fg); }
  img { width:100%; border-radius:4px; display:block; }
  .blk { border:1px solid var(--line); border-radius:6px; padding:10px 12px; margin-bottom:10px; }
  .blk.new { border-color:var(--new); box-shadow:0 0 0 1px var(--new) inset; }
  .blk.new::before { content:"이쪽에만 있음"; display:block; color:var(--new); font-size:12px;
                     font-weight:700; margin-bottom:6px; }
  p { margin:0; font-size:17px; line-height:1.75; }
  h1 { margin:0; font-size:23px; } h2 { margin:0; font-size:19px; }
  table { border-collapse:collapse; width:100%; font-size:13.5px; }
  td, th { border:1px solid var(--line); padding:4px 7px; } th { background:#1b2432; }
</style>
<div class="hd"><div class="t">Pierrot<em>-OCR</em></div><div class="s">__SUB__</div></div>
<div class="cols">__COLS__</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="두 파싱 모드 결과 비교 이미지")
    ap.add_argument("--image", required=True)
    ap.add_argument("--md-a", required=True); ap.add_argument("--label-a", default="A")
    ap.add_argument("--md-b", required=True); ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--col-width", type=int, default=680)
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright

    A, B = md_blocks(a.md_a), md_blocks(a.md_b)
    nA, nB = {norm(x) for x in A}, {norm(x) for x in B}

    def col(label: str, blocks: list[str], other: set) -> str:
        cards = []
        for b in blocks:
            only = norm(b) not in other and len(norm(b)) > 12
            cards.append(f'<div class="blk{" new" if only else ""}">{render_block(b)}</div>')
        return (f'<div class="col"><p class="cap"><b>{html.escape(label)}</b> · '
                f'{len(blocks)}블록</p>{"".join(cards)}</div>')

    cols = (f'<div class="col"><p class="cap"><b>원본</b></p>'
            f'<img src="{img_uri(a.image, a.col_width)}"></div>'
            + col(a.label_a, A, nB) + col(a.label_b, B, nA))
    W = a.col_width * 3 + 120
    doc = (TMPL.replace("__W__", str(W)).replace("__CW__", str(a.col_width + 40))
           .replace("__SUB__", html.escape(a.subtitle or os.path.basename(a.image)))
           .replace("__COLS__", cols))
    tmp = a.out + ".html"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(doc)
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(viewport={"width": W, "height": 1400})
        pg.goto("file://" + os.path.abspath(tmp)); pg.wait_for_timeout(300)
        pg.screenshot(path=a.out, full_page=True, type="jpeg", quality=88)
        br.close()
    os.remove(tmp)
    print(f"[compare] {a.label_a} {len(A)}블록 vs {a.label_b} {len(B)}블록 → {a.out}")


if __name__ == "__main__":
    main()
