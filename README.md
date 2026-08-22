<p align="center">
  <img src="docs/pierrot-vlm-ocr-banner.png" width="100%" alt="PIERROT VLM OCR banner"/>
</p>

<h1 align="center">📄 PIERROT VLM · OCR</h1>

<p align="center">
  <b>PyTorch 스크래치 문서 파싱 VLM — PierrotOCRVLM 추론 배포본</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="pytorch"/>
  <img src="https://img.shields.io/badge/params-0.99B-success.svg" alt="params"/>
  <img src="https://img.shields.io/badge/inference--only-✓-brightgreen.svg" alt="inference-only"/>
  <img src="https://img.shields.io/badge/from--scratch-✓-brightgreen.svg" alt="from-scratch"/>
  <img src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-orange.svg" alt="license"/>
  <img src="https://img.shields.io/badge/commercial%20use-⛔%20prohibited-red.svg" alt="no-commercial"/>
</p>

<p align="center">
  <b>한국어</b> | <a href="README_en.md">English</a>
</p>

---

## 💡 소개

**PIERROT VLM · OCR** 은 [PIERROT VLM](https://github.com/Pierrot-vision/Pierrot-VLM) 에서
학습한 문서 파싱 모델 **PierrotOCRVLM** 을 **돌리기 위한 부분만** 떼어낸 배포본입니다.
페이지 이미지 한 장을 넣으면 **레이아웃 → 영역 인식 → 조립** 을 거쳐
**Markdown / JSON** 이 나옵니다.

> ⚠️ **추론 전용입니다.** 학습 엔트리포인트(`training/`) · 하이퍼파라미터(`args/`) ·
> 학습 엔진(`pierrot/core`) · 데이터 빌더 · 데이터셋 어댑터는 **들어 있지 않습니다.**
> 학습은 원본 저장소에서 합니다 → [이 배포본에 없는 것](#-이-배포본에-없는-것)

모델은 순수 PyTorch 스크래치 구현이고, 알고리즘 골격은 **MinerU2.5** 를 따릅니다 —
태스크별 헤드가 하나도 없고 **체크포인트 하나가 프롬프트만 바꿔** 레이아웃 검출과
영역 인식을 모두 수행하는 coarse-to-fine 2단계입니다.

또한 이곳은 **SOTA를 목표로 하지 않습니다.** 1인 프로젝트의 리소스 안에서
"어디까지 가능한가"를 재는 실험이며, 실패와 오측정까지 [LAB](#-실험-노트-lab) 에
그대로 적어 둡니다.

## 📰 News

- 2026-08-22 — 🩹 **밴드 재검출** — 학습 없이 추론 경로만 고쳐 849쪽 본문 +1.87p (3축 **71.80**)
- 2026-08-20 — 🧩 **조립 수정(body-fill)** — 학습 없이 본문 +11.81p
- 2026-08-20 — 🎓 **V3.4t 티처 증류** — 표·본문 동시 상승
- 2026-08-13 — 🟠 **v3 학습 완료** — KDoc 51쪽 62.4 (표 35.3)
- 2026-08-04 — 🚀 **PierrotOCRVLM 첫 조립** — 하이브리드 이식 게이트 통과(0.986B)

---

## 🧩 모델 한눈에

| | **PierrotOCRVLM** |
|---|---|
| 목적 | 문서 파싱(레이아웃 · 텍스트 · 표 · 수식 · 페이지 통읽기) |
| 알고리즘 골격 | MinerU2.5 — 단일 체크포인트 · 프롬프트 전환 coarse-to-fine |
| 비전 인코더 | 동적해상도 ViT 24층 + **DeepStack**(5/11/17) — Qwen3-VL-2B 이식 |
| 언어 모델 | Qwen3-0.6B 28층 (RMSNorm · QK-Norm · GQA 16/8 · SwiGLU) 이식 |
| 프로젝터 | **패치 머저 4개**(본 1 + DeepStack 3, 출력 1024) — 신규 학습 |
| 위치 인코딩 | **M-RoPE** interleaved `[24, 20, 20]` (head_dim 128) |
| 시퀀스 | ChatML + `<\|vision_start\|><\|image_pad\|>×N<\|vision_end\|>` — **타일 분할 없음** |
| 어텐션 마스크 | 일반 causal |
| 파라미터 | **0.986B** = ViT 306.2M + 머저 83.9M + 디코더 596.0M(embed tie) |
| 가중치 메모리 | ~2 GB (bf16) |
| 이미지 토큰 | 동적 — 인식 ≤2,048 / 레이아웃 ≤1,024 (`max_pixels`/1024) |
| 출력 | 좌표·표·수식 전부 **텍스트 생성**(OCR 특화 헤드 없음) |

> **왜 이 조합인가** — 비전·언어는 검증된 사전학습 부품에서 이식하고, 두 부품을 잇는
> **머저 4개만 새로 학습**한다(modular initialization). ViT 출력이 2048 인데 디코더
> hidden 은 1024 라 머저는 재사용이 불가능하고, 이 랜덤 초기화 경계가 이 모델의
> 최초 학습 리스크였다. 로더가 DeepStack 머저 출력층을 zero-init 해
> 주입을 no-op 에서 시작시킨다.

**아키텍처**

![PierrotOCRVLM 아키텍처](docs/architecture/pierrotocrvlm-architecture.png)

---

## ✅ 검증

- [x] **텍스트 로짓 동등성** — 공식 HF Qwen3-0.6B 대비 max abs diff **1.06e-04** (M-RoPE 교체가 사전학습 언어능력 보존)
- [x] **비전 타워 동등성** — 공식 HF Qwen3-VL ViT hidden 대비 **0.00e+00 (비트 단위 일치)**
- [x] **KV-cache decode == full forward** (소형 랜덤 + 실가중치 멀티모달)
- [x] **849쪽 전량 외부 벤치 재현** (KDoc-OCRBench-V2)

> 위 게이트는 학습 저장소의 `tests/test_pierrotocrvlm.py --full` 에서 돕니다. 이 배포본은
> 같은 모델 코드를 그대로 담되 손실·체크포인팅 등 학습 경로만 뺐습니다.

---

## 🧠 어떻게 동작하나

프롬프트 문자열이 **곧 태스크 스위치**입니다 ([tools/pierrotocr_common.py](tools/pierrotocr_common.py#L20-L39)).

| 프롬프트 | 입력 | 출력 |
|---|---|---|
| `Layout Detection (fine):` | 페이지 축소본(1,024 토큰) | `클래스: x0,y0,x1,y1 ; …` — **나열 순서 = 읽기 순서** (0~999 정규 좌표) |
| `Layout Detection (coarse):` | 〃 | DocLayNet 규약(묶음 분할) |
| `Layout Detection (wild):` | 야외 사진 | 어절 단위 `Text` 박스 |
| `Text Recognition:` | 영역 crop(원본 해상도) | 평문 |
| `Table Recognition:` | 표 crop | **OTSL** → 조립 시 HTML 로 복원 |
| `Formula Recognition:` | 수식 crop | LaTeX |
| `Page Recognition:` | 페이지 전체 | 페이지 Markdown(통읽기 1패스) |

### 파이프라인 — coarse-to-fine 2단계

```
페이지 이미지
   │
   ├─① 레이아웃 패스  축소본(≈1,024 토큰) → "Layout Detection (fine):"
   │                   → [{label, box}] · 나열 순서가 읽기 순서
   │
   ├─② 인식 패스      ①의 box 로 ★원본 해상도★ 에서 crop → 클래스별 프롬프트로
   │                   ★배치 생성★ (Text/Table/Formula, Picture 는 생략)
   │
   └─③ 조립          읽기순서대로 Markdown(표=HTML · 수식=$$…$$) + JSON
```

- **왜 원본에서 crop 하나** — 레이아웃은 축소본에서 읽지만 좌표가 0~999 정규값이라
  원본에 그대로 적용된다. 축소본에서 자르면 작은 글자가 이미 뭉개진 뒤다.
- **왜 배치 생성인가** — 페이지당 요소가 20개 안팎이라 하나씩 돌리면 느리다.
  같은 프롬프트끼리 묶어 우측 패딩 + 어텐션 마스크로 한 번에 생성한다.

### 모드 세 가지

| 모드 | 골격 | 언제 쓰나 |
|---|---|---|
| `coarse-to-fine` (기본) | 레이아웃 지도 | 표·읽기순서가 중요한 문서. 좌표가 함께 나온다 |
| `page` | 페이지 통읽기 1패스 | 단순 조판. 빠르고 본문에 강하다 |
| `hybrid-page` | 통읽기 + c2f 의 표·머리말 이식 | 벤치마크 최고점 경로 (`--body-fill` 동반) |

> **밴드 재검출**(`--relayout-bands`) — 2단 문서에서 레이아웃이 **한쪽 단을 통째로
> 건너뛰는** 실패가 있다. 임계값 문제가 아니라 생성이 그 박스를 **안 낸** 것이라
> 걸러낸 것을 되살릴 방법이 없다. 그런데 **그 영역만 잘라 주면 같은 모델이 정확히
> 찾아낸다.** 그래서 빈 자리를 기하로 찾아 그 부분만 다시 묻는다.

같은 쪽, 고치기 전 — 오른쪽 단 위쪽에 박스가 하나도 없다(두 문단이 결과에서 사라졌다):

![밴드 재검출 전](docs/images/pierrotocrvlm/band-relayout/p02-before.jpg)

밴드 재검출 뒤 — 우단 상단에 박스가 생기고 문단이 제자리에 들어왔다:

![밴드 재검출 후](docs/images/pierrotocrvlm/band-relayout/p02-band-recovered.jpg)

---

## 📊 성능 — KDoc-OCRBench-V2 (849쪽 전량)

한국어 문서 벤치마크. 공식 리더보드 모델과 **같은 표**에 둡니다.

| 순위 | 모델 | Header/Footer | Long Text | Table | 3축 평균 | 채점 |
|---:|---|---:|---:|---:|---:|---|
| 1 | BizOnAI-OCR | 94.7 | **77.9** | **58.1** | **76.9** | 정식 |
| **2** | **PierrotOCRVLM (V3 + body-fill + 밴드)** | 96.09 | **76.61** | 42.70 | **71.80** | 공백무시 |
| 3 | PierrotOCRVLM (V3 + body-fill) | 96.72 | 74.74 | 42.42 | 71.29 | 공백무시 |
| 4 | PaddleOCR-VL | 95.6 | 66.2 | 48.9 | 70.2 | 정식 |
| 5 | DeepSeek OCR | 95.8 | 64.5 | 46.6 | 69.0 | 정식 |
| 6 | olmOCR v0.2.0 | 95.2 | 65.0 | 44.9 | 68.4 | 정식 |
| 7 | PierrotOCRVLM (V3 기준선) | 96.97 | 62.93 | 42.42 | 67.44 | 공백무시 |
| 8 | GLM-4.1V-OCR | **97.4** | 52.9 | 30.0 | 60.1 | 정식 |

> ⚠️ **직접 비교가 아닙니다.** 우리 값은 **공백무시** 채점, 외부는 **정식** 채점이라
> 우리 쪽이 유리한 자입니다(50쪽 실측: PaddleOCR-VL 정식 53.71 → 공백무시 59.11).
> 공백무시를 기본으로 쓰는 이유는 학습셋 일부가 표 셀의 공백을 잃은 채로 들어갔기
> 때문이며, 그 경위는 [실험 노트 §8](LAB/pierrotocrvlm/pierrotocrvlm.md#8-성능) 에 있습니다.

본문은 1위와 **1.3p** 차까지 좁혔고, **남은 격차는 표 하나**입니다(42.70 vs 58.1).
2 · 3 · 7위의 차이는 **재학습이 아니라 추론·조립 경로 수정만으로** 얻은 것입니다
(기준선 67.44 → 71.80, **+4.36p**).

**예측 예시** (왼쪽=모델 입력, 오른쪽=모델 생성 결과)

| 레이아웃 | 표(OTSL→HTML) |
|---|---|
| ![레이아웃 예측](docs/images/pierrotocrvlm/predictions/v2/sbs_layout_ccpdf.png) | ![표 예측](docs/images/pierrotocrvlm/predictions/v2/sbs_table_real.png) |

| 페이지 통읽기(한국어) | 야외 텍스트(wild) |
|---|---|
| ![페이지 통읽기](docs/images/pierrotocrvlm/predictions/v2/sbs_page_ko.png) | ![야외 텍스트](docs/images/pierrotocrvlm/predictions/v2/sbs_text_wild.png) |

---

## 📦 설치

```bash
# 1) conda 환경 생성 · 활성화
conda create -n pierrot-ocr python=3.10 -y
conda activate pierrot-ocr

# 2) 의존성 설치
cd Pierrot_VLM_OCR
pip install -r requirements.txt
```

의존성은 여섯 개뿐입니다 — `torch` · `transformers`(토크나이저 로드 전용) ·
`safetensors` · `huggingface_hub` · `pillow` · `numpy`.
모델 구현이 스크래치(raw config 파싱)라 **특정 transformers 버전이 필요 없습니다.**
학습용 `accelerate` 는 들어가지 않습니다.

> GPU 환경에서는 CUDA 버전에 맞는 PyTorch 빌드를 먼저 설치하는 것을 권장합니다.
> bf16 기준 가중치가 ~2 GB 라 8 GB 급 GPU 에서도 페이지 파싱이 돕니다.

---

## 🔮 추론

### 페이지 한 장 — `infer/infer_pierrotocrvlm.py`

```bash
# 기본: coarse-to-fine 2단계 → Markdown + JSON
python infer/infer_pierrotocrvlm.py --model ./outputs/pierrotocrvlm_v3/final \
    --image page.png --out results/parsed --save-viz

# 페이지 통읽기 1패스(단순 조판에 빠르고 강하다)
python infer/infer_pierrotocrvlm.py --model ./outputs/pierrotocrvlm_v3/final \
    --image page.png --mode page

# 야외 사진(간판·표지판): 어절 박스 규약으로 검출 → crop 인식
python infer/infer_pierrotocrvlm.py --model ./outputs/pierrotocrvlm_v3/final \
    --image photo.jpg --layout-prompt wild
```

주요 인자:

| 인자 | 기본 | 설명 |
|---|---|---|
| `--mode` | `coarse-to-fine` | `page` 로 통읽기 1패스 |
| `--layout-prompt` | `fine` | `coarse`(DocLayNet 규약) / `wild`(야외 어절) |
| `--batch-size` | 8 | crop 병렬 생성 배치 |
| `--max-new-tokens` | 1024 | 본문 crop 생성 상한 |
| `--table-max-new-tokens` | 4096 | **표만 따로** — 통계표는 셀이 수백 개다 |
| `--stop-on-cycle` | 32 | 퇴행 반복 차단(0=끔). 점수는 그대로인데 출력이 절반으로 준다 |
| `--save-viz` | off | 레이아웃 박스를 원본에 그려 저장(검수용) |
| `--dtype` | `bfloat16` | `float32` / `float16` |

산출물: `<stem>.md`(읽기순서 Markdown) · `<stem>.json`(요소별 클래스·박스·텍스트) ·
`<stem>_layout.jpg`(`--save-viz`).

### 여러 장 배치 — `benchmark/run_pages.py`

한 장짜리 CLI 로 수백 장을 돌리면 모델을 매번 다시 올립니다(장당 20초 낭비).
배치 러너는 모델을 한 번만 올리고 목록을 훑습니다. **추론 로직은 위 파일의 함수를
그대로 가져다 씁니다** — 벤치마크 때문에 파싱 규칙이 갈라지면 점수가 실제와 달라집니다.

```bash
# 벤치마크 최고점 경로(849쪽 3축 71.80 을 낸 조합)
python benchmark/run_pages.py --model ./outputs/pierrotocrvlm_v3/final \
    --images "/path/pages/*.jpg" --out results/md \
    --mode hybrid-page --body-fill --relayout-bands

# GPU 를 나눠 쓰려면 샤드로 쪼개 여러 프로세스로 띄운다
CUDA_VISIBLE_DEVICES=0 python benchmark/run_pages.py ... --shard 0 --num-shards 4 &
CUDA_VISIBLE_DEVICES=1 python benchmark/run_pages.py ... --shard 1 --num-shards 4 &
```

| 인자 | 설명 |
|---|---|
| `--mode` | `coarse-to-fine` / `page` / `hybrid` / `hybrid-page` |
| `--body-fill` | `hybrid-page` 가 버리는 본문 crop 을 끝에 붙인다 (**+3.85p**) |
| `--relayout-bands` | 레이아웃이 통째로 건너뛴 밴드만 다시 검출 (**본문 +1.87p**) |
| `--table-pad` | 표 crop 비율 패딩(0.02 = 2%). 실측 최적 2% |
| `--page-no-repeat-ngram` | 통읽기 전용 n-gram 금지. **표·crop 에는 걸지 않는다** |
| `--save-layout` / `--save-trace` | 레이아웃 원시 출력 · crop 별 무절단 트레이스 저장(진단) |
| `--shard` / `--num-shards` | 다중 GPU 분할 |

> ⚠️ **모드를 잘못 고르면 옵션 효과가 0 입니다.** `--relayout-bands` 를 `hybrid-page`
> **단독**으로 켜면 밴드를 47개 찾고도 점수가 소수점까지 같습니다 — 그 모드는
> 통읽기를 뼈대로 써서 새로 찾은 본문이 흘러갈 통로가 없습니다.
> `coarse-to-fine` 골격이거나 `--body-fill` 과 함께일 때만 값어치가 나옵니다.

### 파이썬에서 직접

```python
from PIL import Image
from pierrot.models.pierrotocrvlm import load_pretrained
from tools.pierrotocr_common import PROMPT_LAYOUT_FINE, PROMPT_TABLE

model, processor = load_pretrained("./outputs/pierrotocrvlm_v3/final", device="cuda")
model.eval()

enc   = processor([Image.open("page.png").convert("RGB")], [PROMPT_LAYOUT_FINE])
enc   = {k: v.to("cuda") for k, v in enc.items()}
out   = model.generate(enc["input_ids"], enc["pixel_values"], enc["image_grid_thw"],
                       enc["attention_mask"], max_new_tokens=768,
                       eos_token_id=processor.eos_token_id)
print(processor.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

전처리 예산(`min_pixels` / `max_pixels` / `layout_max_pixels`)은 학습 산출물의
sidecar(`pierrotocrvlm_preprocessor.json`)에서 **자동 복원**됩니다 — 학습과 추론의
이미지 토큰 수가 어긋나지 않게 하는 장치이므로 특별한 이유 없이 덮어쓰지 마세요.

---

## 🧾 출력 형식

**Markdown** — 읽기순서대로 조립. 클래스별 규칙:

| 클래스 | 출력 |
|---|---|
| `Title` / `Section-header` | `# ` / `## ` 접두 |
| `Text` / `Footnote` | 평문 |
| `List-item` | `- ` 접두 |
| `Caption` | `*이탤릭*` |
| `Table` | OTSL → **HTML 표** 복원 |
| `Formula` | `$$ … $$` |
| `Picture` | `![Picture]()` 자리표시자(인식 생략) |
| `Page-header` / `Page-footer` | 본문에서 제외 |

**JSON** — `{image, size, elements: [{label, box, text}]}`.
`box` 는 0~999 정규 좌표라 원본 크기와 무관하게 재사용할 수 있습니다.

**OTSL** — 표 직렬화 형식([tools/otsl.py](tools/otsl.py)). HTML 대비 토큰이 절반 가까이
줄고 태그 미폐합이 문법적으로 불가능합니다. `otsl_to_html()` 로 되돌립니다.

---

## 📁 디렉토리 구조

```
Pierrot_VLM_OCR/
├── infer/
│   └── infer_pierrotocrvlm.py  # ★ 페이지 1장 CLI — 2단계 파싱 · 조립 · 시각화
├── benchmark/
│   └── run_pages.py            # ★ 배치 러너 — 모델 1회 로드 · 모드/샤드/진단 옵션
├── eval/
│   └── metrics_ocr.py          # 레이아웃 출력 파서(parse_layout) + 지표(NED · TEDS · F1 · tau)
├── tools/
│   ├── pierrotocr_common.py    # 태스크 프롬프트 단일 소스 (= 태스크 스위치)
│   ├── otsl.py                 # OTSL ↔ HTML 표 변환기
│   ├── hybrid_merge.py         # page ↔ c2f 병합 · 단 분할 · 미덮 밴드 탐지
│   ├── make_demo_viewer.py     # 3패널 재생 뷰어(원본+RAW 스트림+재조판) HTML
│   ├── compare_modes.py        # 두 모드 결과 나란히 비교 이미지
│   └── record_demo_gif.py      # 뷰어 → 데모 GIF 녹화 (playwright 필요)
├── requirements.txt
├── LAB/pierrotocrvlm/          # 실험 노트 1편 — 설계·학습·실험·결과·실패 (아래 참조)
├── docs/                       # 아키텍처 그림 · LAB 그림 · 배너
└── pierrot/
    └── models/pierrotocrvlm/   # 알고리즘 패키지 (추론 경로만)
        ├── config.py           #   PierrotOCRConfig / Text / Vision (HF config.json 과 1:1)
        ├── modeling/
        │   ├── vision.py       #     동적해상도 ViT + bilinear 위치보간 + 패치머저 + DeepStack
        │   ├── text.py         #     Qwen3 디코더 (RMSNorm+M-RoPE+QK-Norm+GQA+SwiGLU, +KVCache)
        │   └── pierrotocrvlm.py#     병합(masked_scatter) + M-RoPE 위치 + generate(사이클/n-gram 차단)
        ├── processor.py        #   동적해상도 패치화(smart_resize) + ChatML 프롬프트 인코딩
        ├── detection.py        #   검출/레이아웃 출력 파싱 & 시각화
        └── weights.py          #   체크포인트 로드(config + safetensors/model.pt + sidecar)
```

---

## 🧪 실험 노트 (LAB)

설계·학습 방법·실험·결과를 **한 문서**에 정리해 두었습니다 —
[**LAB/pierrotocrvlm/pierrotocrvlm.md**](LAB/pierrotocrvlm/pierrotocrvlm.md).

성공한 것만 적지 않았습니다. **기각된 가설 15개**와 **계측 결함 13건**이 함께 들어 있고,
이 프로젝트에서 가장 비쌌던 것이 그쪽입니다.

| 절 | 내용 |
|---|---|
| [1 개요](LAB/pierrotocrvlm/pierrotocrvlm.md#1-개요) · [2 아키텍처](LAB/pierrotocrvlm/pierrotocrvlm.md#2-아키텍처) | MinerU2.5 골격에서 무엇을 바꿨나 · 기술 계보 · 머저 4개가 최대 리스크인 이유 |
| [3 태스크](LAB/pierrotocrvlm/pierrotocrvlm.md#3-태스크-설계) · [4 추론](LAB/pierrotocrvlm/pierrotocrvlm.md#4-추론-파이프라인) | 프롬프트 = 태스크 스위치 · 검출 규약 3분리 · OTSL · 조립 +11.81p · 밴드 재검출 |
| [5 학습](LAB/pierrotocrvlm/pierrotocrvlm.md#5-학습-방법) · [6 데이터](LAB/pierrotocrvlm/pierrotocrvlm.md#6-데이터) | Stage 0~3 커리큘럼 · **배합은 토큰으로** · 데이터에서 배운 것과 사고 3건 |
| [7 실험](LAB/pierrotocrvlm/pierrotocrvlm.md#7-실험-기록) · [8 성능](LAB/pierrotocrvlm/pierrotocrvlm.md#8-성능) | v1 → v2 → v3 → 3연패 → 티처 증류 → V3.5/3.6 · 리더보드 대조 |
| [9 기각된 가설](LAB/pierrotocrvlm/pierrotocrvlm.md#9-기각된-가설) · [10 계측 결함](LAB/pierrotocrvlm/pierrotocrvlm.md#10-계측-결함-13건) · [11 남은 병목](LAB/pierrotocrvlm/pierrotocrvlm.md#11-남은-병목) | 다시 하지 말 것 · 도구부터 의심할 것 · 단계별 생존 분해 |

문서에서 `args/` · `training/` · `tools/build_*` 를 언급하는 대목은 **학습 저장소**를
가리킵니다 — 이 배포본에는 없는 파일들입니다.

---

## 🚫 이 배포본에 없는 것

| 없는 것 | 어디에 있나 |
|---|---|
| 학습 엔트리포인트 `training/train_pierrotocrvlm.py` | 학습 저장소 |
| 하이퍼파라미터 단일 소스 `args/pierrotocrvlm.py` | 〃 |
| Accelerate 학습 엔진 `pierrot/core` (Trainer · 스케줄러 · 레지스트리) | 〃 |
| 데이터셋 어댑터 · collate · 증강 (`dataset.py` · `pierrot/data`) | 〃 |
| 데이터 빌더 `tools/build_*_jsonl.py` (30여 개) | 〃 |
| 부품 이식 조립 `weights.load_hybrid()` (학습 시작점) | 〃 |
| val 셋 평가 하네스 `eval/eval_pierrotocrvlm.py` | 〃 |

모델 코드(`config` · `modeling` · `processor` · `weights`)는 학습 저장소와 **같은 구현**이며,
손실 계산 · 활성화 체크포인팅 · 옵티마이저 파라미터 그룹 · 라벨 생성 등
**학습에서만 쓰는 경로만** 제거했습니다. 체크포인트 호환성은 그대로입니다.

---

## 📚 참조 (Reference)

관련 논문 정리 · 리뷰 → [Pierrot-vision/Reading-Papers — VLM](https://github.com/Pierrot-vision/Reading-Papers#-vlm)

- **MinerU2.5** — coarse-to-fine 2단계 문서 파싱 골격
- **Qwen3-VL** / **Qwen3** — 비전 타워 · 언어 디코더 부품
- **OTSL** — 표 구조 직렬화
- **KDoc-OCRBench-V2** · **OmniDocBench** · **CC-OCR** — 평가 벤치마크

---

## 📄 라이선스

이 프로젝트(코드 · 문서)는 **CC BY-NC-SA 4.0** 을 따릅니다 — 출처 표기 · 비영리 · 동일조건 변경허락. 자세한 내용은 [LICENSE](LICENSE) 참고. (사용된 서드파티 데이터셋 · 모델 · 라이브러리는 각자의 라이선스를 따릅니다.)

> ⛔ **상업적 사용 금지 (NonCommercial)** — 비영리 목적으로만 사용할 수 있습니다. 상업적 이용이 필요하면 별도 문의 바랍니다.

---

## 📮 문의

- [메일](mailto:peternara@naver.com) 또는 [GitHub Issue](https://github.com/Pierrot-vision/Pierrot-VLM/issues) 를 통해 관련 질문·문의 부탁드립니다. 대답할수 있는 내용이라면 성실이 답변드리겠습니다.
- 참고로, 이미 GitHub(README · 코드 · 문서)에 있는 내용을 다시 문의하시면 답을 드리지 못할 수 있는 점 양해 부탁드립니다.
