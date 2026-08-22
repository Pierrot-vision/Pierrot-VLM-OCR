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
> 모델 코드는 학습 저장소와 같은 구현이고 학습 전용 경로만 제거했으므로 체크포인트는 그대로 호환됩니다.

## 📰 News

- 2026-08-22 — 🏆 **KDoc-OCRBench-V2 2위** — 상업용 모델을 빼면 **1위** (3축 평균 71.80)
- 2026-08-22 — 🚀 **추론 코드 공개**

---

## 📊 성능 — KDoc-OCRBench-V2 (849쪽 전량)

| 순위 | 모델 | Header/Footer | Long Text | Table | 3축 평균 |
|---:|---|---:|---:|---:|---:|
| 1 | BizOnAI-OCR *(상업용)* | 94.7 | **77.9** | **58.1** | **76.9** |
| **2** | **Ours** | 96.09 | 76.61 | 42.70 | 71.80 |
| 3 | PaddleOCR-VL | 95.6 | 66.2 | 48.9 | 70.2 |
| 4 | DeepSeek OCR | 95.8 | 64.5 | 46.6 | 69.0 |
| 5 | olmOCR v0.2.0 | 95.2 | 65.0 | 44.9 | 68.4 |
| 6 | GLM-4.1V-OCR | **97.4** | 52.9 | 30.0 | 60.1 |

---

## 🎬 데모

3패널 재생 뷰어([tools/make_demo_viewer.py](tools/make_demo_viewer.py))로 캡처한 실제 파싱 과정입니다 —
왼쪽 원본 + 레이아웃 박스, 가운데 모델 원시 출력, 오른쪽 재조판 결과.

**2단 조판 학술 논문**

![2단 조판 논문 파싱 데모](https://github.com/Pierrot-vision/Pierrot-VLM-OCR/releases/download/v0.1.0/demo_paper.gif)

**공공 보고서**

![공공 보고서 파싱 데모](https://github.com/Pierrot-vision/Pierrot-VLM-OCR/releases/download/v0.1.0/demo_kdi.gif)

> 재생 순서는 우리가 정한 것이고, 좌표와 내용은 모델이 낸 값 그대로입니다 —
> 이 모델은 1패스 인터리브가 아니라 coarse-to-fine 2패스라 중앙 스트림은
> **두 패스의 실제 출력을 읽기순서로 엮은 것**입니다.
> (각 28~32MB 라 로딩에 시간이 걸립니다.)

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

## 📚 참조 (Reference)

관련 논문 정리 · 리뷰 → [Pierrot-vision/Reading-Papers — VLM](https://github.com/Pierrot-vision/Reading-Papers#-vlm)

- **KDoc-OCRBench-V2** · **OmniDocBench** · **CC-OCR** — 평가 벤치마크

---

## 📄 라이선스

이 프로젝트(코드 · 문서)는 **CC BY-NC-SA 4.0** 을 따릅니다 — 출처 표기 · 비영리 · 동일조건 변경허락. 자세한 내용은 [LICENSE](LICENSE) 참고. (사용된 서드파티 데이터셋 · 모델 · 라이브러리는 각자의 라이선스를 따릅니다.)

> ⛔ **상업적 사용 금지 (NonCommercial)** — 비영리 목적으로만 사용할 수 있습니다. 상업적 이용이 필요하면 별도 문의 바랍니다.

---

## 📮 문의

- [메일](mailto:peternara@naver.com) 또는 [GitHub Issue](https://github.com/Pierrot-vision/Pierrot-VLM/issues) 를 통해 관련 질문·문의 부탁드립니다. 대답할수 있는 내용이라면 성실이 답변드리겠습니다.
- 참고로, 이미 GitHub(README · 코드 · 문서)에 있는 내용을 다시 문의하시면 답을 드리지 못할 수 있는 점 양해 부탁드립니다.
