#!/usr/bin/env python
"""3패널 재생 뷰어(HTML) → 데모 GIF 녹화.

`tools/make_demo_viewer.py` 가 만든 뷰어를 헤드리스 크로미움으로 열어 **재생을 직접
한 틱씩 몰면서** 프레임을 캡처한다. 실시간 재생을 옆에서 찍으면 캡처 지연(장당
100~200ms)만큼 화면이 건너뛰어 타이핑이 뚝뚝 끊긴다 — 그래서 타이머를 쓰지 않고
`tick()` 을 호출한 직후에 찍는다. 프레임과 내용이 1:1 로 맞는다.

브라우저는 playwright 가 설치된 별도 파이썬(가상환경)에서 돌린다. 이 스크립트는
그 환경의 인터프리터로 실행한다:

    <pwenv>/bin/python tools/record_demo_gif.py \\
        --html results/demo_paper/viewer.html \\
        --out results/demo_paper/demo.gif --frames 240 --width 1280
"""

from __future__ import annotations

import argparse
import glob
import os

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

# 재생 중 블록이 바뀔 때마다 부드러운 스크롤이 걸리면 프레임이 흐르는 도중에 찍힌다.
# 녹화에서는 즉시 이동으로 바꿔 프레임마다 화면이 확정되게 한다.
_INIT_JS = """
const _orig = Element.prototype.scrollIntoView;
Element.prototype.scrollIntoView = function (o) { _orig.call(this, {block: "center"}); };
"""


def capture(html: str, frames: int, vw: int, vh: int, speed: str,
            hold_head: int, hold_tail: int, hold_page: int = 0):
    """뷰어를 열어 프레임 목록(PIL 이미지)을 만든다."""
    url = "file://" + os.path.abspath(html)
    shots = []
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(viewport={"width": vw, "height": vh})
        pg.add_init_script(_INIT_JS)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url)
        pg.wait_for_timeout(800)
        pg.select_option("#speed", speed)

        # 전체 틱 수를 미리 세어 프레임당 몇 틱을 몰지 정한다(길이에 상관없이 같은 재생 시간).
        total = pg.evaluate(
            "s => queue.reduce((a, q) => a + Math.ceil((q.el.raw||'').length / s) + 1, 0)",
            int(speed))
        per = max(1, -(-total // max(1, frames - hold_head - hold_tail)))
        print(f"[gif] 블록 {pg.evaluate('queue.length')}개 · 총 {total}틱 · 프레임당 {per}틱")

        def shot():
            # ★ 프레임은 항상 JPEG 로 찍는다 — PNG 는 캡처가 느리고 캐시가 몇 배로 커진다.
            #   어차피 GIF 는 256색으로 양자화되므로 무손실을 유지할 이유가 없다.
            shots.append(Image.open(__import__("io").BytesIO(
                pg.screenshot(type="jpeg", quality=90))).convert("RGB"))

        for _ in range(hold_head):                      # 시작 화면을 잠깐 보여 준다
            shot()
        # 버튼 상태만 재생 중으로 바꾸고(라벨 일관성) 타이머는 쓰지 않는다.
        pg.evaluate("document.getElementById('play').textContent = '❚❚ STOP'")
        prev_pi = 0
        while pg.evaluate("qi") < pg.evaluate("queue.length"):
            pg.evaluate(f"() => {{ for (let i = 0; i < {per}; i++) tick(); }}")
            shot()
            if hold_page:
                cur_pi = pg.evaluate("qi < queue.length ? queue[qi].pi : -1")
                if cur_pi != prev_pi:                    # 쪽이 넘어갔다
                    pg.evaluate(f"() => {{ const w = document.querySelectorAll"
                                f"('.pagewrap')[{prev_pi}]; if (w) w.scrollIntoView"
                                f"({{block:'center'}}); }}")
                    for _ in range(hold_page):
                        shot()
                    prev_pi = cur_pi if cur_pi >= 0 else prev_pi
            if len(shots) > frames * 3:                 # 안전장치
                break
        for _ in range(hold_tail):                      # 완성 화면을 붙잡아 둔다
            shot()
        if errs:
            print("[gif] ★ 페이지 오류:", errs[:3])
        br.close()
    return shots


def main() -> None:
    ap = argparse.ArgumentParser(description="뷰어 HTML → 데모 GIF")
    ap.add_argument("--html", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=240, help="목표 프레임 수")
    ap.add_argument("--width", type=int, default=1280, help="GIF 폭(px)")
    ap.add_argument("--viewport", default="1920x1000")
    ap.add_argument("--speed", default="25", help="뷰어 속도 옵션값(틱당 문자 수)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--hold-head", type=int, default=6)
    ap.add_argument("--hold-tail", type=int, default=18)
    # ★ 페이지를 다 읽은 순간 화면은 이미 다음 쪽으로 스크롤돼 있다. 그래서 "이 쪽을
    #   이만큼 잡아냈다"는 완성 장면이 GIF 에 거의 남지 않는다(실측: 2쪽 오른쪽 단이
    #   켜지자마자 3쪽으로 넘어감). 쪽이 끝날 때마다 그 쪽 전체를 보여주며 멈춘다.
    ap.add_argument("--hold-page", type=int, default=10,
                    help="한 쪽을 다 읽었을 때 그 쪽 전체를 보여주며 멈출 프레임 수")
    ap.add_argument("--frame-dir", default="", help="캡처 프레임 PNG 캐시(있으면 재사용)")
    a = ap.parse_args()

    vw, vh = (int(x) for x in a.viewport.split("x"))
    cached = sorted(glob.glob(os.path.join(a.frame_dir, "f*.jpg"))) if a.frame_dir else []
    if cached:
        shots = [Image.open(f).convert("RGB") for f in cached]
        print(f"[gif] 캐시 프레임 {len(shots)}장 재사용 ({a.frame_dir})")
    else:
        shots = capture(a.html, a.frames, vw, vh, a.speed, a.hold_head, a.hold_tail,
                        a.hold_page)
        print(f"[gif] 프레임 {len(shots)}장 캡처")
        if a.frame_dir:
            os.makedirs(a.frame_dir, exist_ok=True)
            for i, s_ in enumerate(shots):
                s_.save(os.path.join(a.frame_dir, f"f{i:04d}.jpg"), quality=90)

    # 폭을 줄이고 팔레트를 공통으로 잡는다 — 프레임마다 팔레트가 다르면 색이 깜빡인다.
    if a.width and a.width < vw:
        h = round(vh * a.width / vw)
        shots = [s.resize((a.width, h), Image.LANCZOS) for s in shots]
    pal = shots[len(shots) // 2].quantize(colors=a.colors, method=Image.MEDIANCUT)
    conv = [s.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for s in shots]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # ★ 연속으로 같은 프레임은 **우리가 직접 합치고 그만큼 지속시간을 늘린다.**
    #   PIL 은 중복 프레임을 지우면서 시간을 합쳐 주지 않아, 쪽 완성 장면의 멈춤이
    #   통째로 사라진다(실측: 213프레임 캡처 → 136프레임, 정지 구간 0).
    base = round(1000 / a.fps)
    merged, durs = [conv[0]], [base]
    for prev_rgb, img, cur_rgb in zip(shots, conv[1:], shots[1:]):
        if ImageChops.difference(cur_rgb, prev_rgb).getbbox() is None:
            durs[-1] += base
        else:
            merged.append(img)
            durs.append(base)
    merged[0].save(a.out, save_all=True, append_images=merged[1:],
                   duration=durs, loop=0, optimize=True, disposal=1)
    conv = merged
    print(f"[gif] → {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB, "
          f"{len(conv)}프레임 · {a.fps}fps · {len(conv)/a.fps:.0f}초)")


if __name__ == "__main__":
    main()
