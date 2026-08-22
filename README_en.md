<p align="center">
  <img src="docs/pierrot-vlm-ocr-banner.png" width="100%" alt="PIERROT VLM OCR banner"/>
</p>

<h1 align="center">📄 PIERROT VLM · OCR</h1>

<p align="center">
  <b>A from-scratch PyTorch document-parsing VLM — PierrotOCRVLM inference distribution</b>
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
  <a href="README.md">한국어</a> | <b>English</b>
</p>

---

## 💡 Introduction

**PIERROT VLM · OCR** carves out only the parts needed to **run** **PierrotOCRVLM**, the
document-parsing model trained in [PIERROT VLM](https://github.com/Pierrot-vision/Pierrot-VLM).
Feed it a page image and it goes **layout → region recognition → assembly**, producing
**Markdown / JSON**.

> ⚠️ **Inference only.** Training entrypoints (`training/`), hyperparameters (`args/`),
> the training engine (`pierrot/core`), data builders, and dataset adapters are **not
> included.** Training lives in the original repository →
> [What is not here](#-what-is-not-here)

The model is a pure-PyTorch from-scratch implementation, and the algorithmic skeleton
follows **MinerU2.5** — there is no task-specific head at all; **a single checkpoint
switches roles by prompt alone** in a coarse-to-fine two-pass pipeline.

This project also **does not chase SOTA.** It measures how far a one-person project can
get within its resources, and failures and mismeasurements are written down as-is in
[LAB](#-lab-notes).

## 📰 News

- 2026-08-22 — 🚀 **Inference code released**

---

## 📊 Performance — KDoc-OCRBench-V2 (all 849 pages)

A Korean document benchmark. Shown in **the same table** as the official leaderboard.

| Rank | Model | Header/Footer | Long Text | Table | 3-axis mean | Scoring |
|---:|---|---:|---:|---:|---:|---|
| 1 | BizOnAI-OCR | 94.7 | **77.9** | **58.1** | **76.9** | official |
| **2** | **PierrotOCRVLM (V3 + body-fill + bands)** | 96.09 | **76.61** | 42.70 | **71.80** | space-insensitive |
| 3 | PierrotOCRVLM (V3 + body-fill) | 96.72 | 74.74 | 42.42 | 71.29 | space-insensitive |
| 4 | PaddleOCR-VL | 95.6 | 66.2 | 48.9 | 70.2 | official |
| 5 | DeepSeek OCR | 95.8 | 64.5 | 46.6 | 69.0 | official |
| 6 | olmOCR v0.2.0 | 95.2 | 65.0 | 44.9 | 68.4 | official |
| 7 | PierrotOCRVLM (V3 baseline) | 96.97 | 62.93 | 42.42 | 67.44 | space-insensitive |
| 8 | GLM-4.1V-OCR | **97.4** | 52.9 | 30.0 | 60.1 | official |

> ⚠️ **Not a direct comparison.** Our numbers use **space-insensitive** scoring while the
> external ones are **official**, which favors us (measured on 50 pages: PaddleOCR-VL
> official 53.71 → space-insensitive 59.11). We default to space-insensitive because part
> of the training set lost table-cell spacing; the full account is in
> [the LAB note §8](LAB/pierrotocrvlm/pierrotocrvlm.md#8-성능).

Body text is now within **1.3p** of first place, and **the remaining gap is tables alone**
(42.70 vs 58.1). The difference between ranks 2, 3 and 7 came **purely from inference and
assembly fixes, not retraining** (baseline 67.44 → 71.80, **+4.36p**).

**Prediction examples** (left = model input, right = generated output)

| Layout | Table (OTSL→HTML) |
|---|---|
| ![layout prediction](docs/images/pierrotocrvlm/predictions/v2/sbs_layout_ccpdf.png) | ![table prediction](docs/images/pierrotocrvlm/predictions/v2/sbs_table_real.png) |

| Whole-page reading (Korean) | Wild text |
|---|---|
| ![page reading](docs/images/pierrotocrvlm/predictions/v2/sbs_page_ko.png) | ![wild text](docs/images/pierrotocrvlm/predictions/v2/sbs_text_wild.png) |

---

## 📦 Installation

```bash
# 1) create and activate a conda environment
conda create -n pierrot-ocr python=3.10 -y
conda activate pierrot-ocr

# 2) install dependencies
cd Pierrot_VLM_OCR
pip install -r requirements.txt
```

There are only six dependencies — `torch` · `transformers` (tokenizer loading only) ·
`safetensors` · `huggingface_hub` · `pillow` · `numpy`. The model implementation is
from scratch (raw config parsing), so **no specific transformers version is required**.
`accelerate` is not needed since training is not included.

> On GPU, install a PyTorch build matching your CUDA version first. Weights are ~2 GB in
> bf16, so page parsing runs even on an 8 GB-class GPU.

---

## 🔮 Inference

### One page — `infer/infer_pierrotocrvlm.py`

```bash
# default: coarse-to-fine two-pass → Markdown + JSON
python infer/infer_pierrotocrvlm.py --model ./outputs/pierrotocrvlm_v3/final \
    --image page.png --out results/parsed --save-viz

# single-pass whole-page reading (fast and strong on simple layouts)
python infer/infer_pierrotocrvlm.py --model ./outputs/pierrotocrvlm_v3/final \
    --image page.png --mode page

# outdoor photos (signs, storefronts): word-box detection → crop recognition
python infer/infer_pierrotocrvlm.py --model ./outputs/pierrotocrvlm_v3/final \
    --image photo.jpg --layout-prompt wild
```

Main arguments:

| Argument | Default | Description |
|---|---|---|
| `--mode` | `coarse-to-fine` | `page` for single-pass reading |
| `--layout-prompt` | `fine` | `coarse` (DocLayNet convention) / `wild` (outdoor words) |
| `--batch-size` | 8 | Crops generated in parallel |
| `--max-new-tokens` | 1024 | Generation budget for body crops |
| `--table-max-new-tokens` | 4096 | **Tables only** — statistical tables have hundreds of cells |
| `--stop-on-cycle` | 32 | Degenerate-repetition cutoff (0 = off). Scores stay identical while output halves |
| `--save-viz` | off | Draw layout boxes on the original for review |
| `--dtype` | `bfloat16` | `float32` / `float16` |

Outputs: `<stem>.md` (Markdown in reading order) · `<stem>.json` (per-element class, box,
text) · `<stem>_layout.jpg` (with `--save-viz`).

### Many pages — `benchmark/run_pages.py`

Running hundreds of pages through the single-page CLI reloads the model every time
(~20 s wasted per page). The batch runner loads once and walks the list. **It imports the
inference functions from that same file** — if parsing rules forked for benchmarking, the
scores would stop matching the real thing.

```bash
# the best benchmark path (the combination that produced 71.80 over 849 pages)
python benchmark/run_pages.py --model ./outputs/pierrotocrvlm_v3/final \
    --images "/path/pages/*.jpg" --out results/md \
    --mode hybrid-page --body-fill --relayout-bands

# split across GPUs with shards
CUDA_VISIBLE_DEVICES=0 python benchmark/run_pages.py ... --shard 0 --num-shards 4 &
CUDA_VISIBLE_DEVICES=1 python benchmark/run_pages.py ... --shard 1 --num-shards 4 &
```

| Argument | Description |
|---|---|
| `--mode` | `coarse-to-fine` / `page` / `hybrid` / `hybrid-page` |
| `--body-fill` | Append the body crops `hybrid-page` would discard (**+3.85p**) |
| `--relayout-bands` | Re-detect only the bands the layout pass skipped (**body text +1.87p**) |
| `--table-pad` | Ratio padding for table crops (0.02 = 2%). Measured optimum is 2% |
| `--page-no-repeat-ngram` | n-gram ban for whole-page reading only — **never applied to tables/crops** |
| `--save-layout` / `--save-trace` | Save raw layout output / untruncated per-crop traces (diagnostics) |
| `--shard` / `--num-shards` | Multi-GPU split |

> ⚠️ **Pick the wrong mode and an option does nothing.** Turning on `--relayout-bands`
> with `hybrid-page` **alone** finds 47 bands and leaves the score identical to the
> decimal — that mode uses whole-page reading as its skeleton, so newly found body text
> has no channel to flow through. Bands pay off only on a `coarse-to-fine` skeleton or
> together with `--body-fill`.

### From Python

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

Preprocessing budgets (`min_pixels` / `max_pixels` / `layout_max_pixels`) are **restored
automatically** from the checkpoint sidecar (`pierrotocrvlm_preprocessor.json`). They keep
the image-token count identical between training and inference, so do not override them
without a specific reason.

---

## 🧾 Output format

**Markdown** — assembled in reading order, per class:

| Class | Output |
|---|---|
| `Title` / `Section-header` | `# ` / `## ` prefix |
| `Text` / `Footnote` | Plain text |
| `List-item` | `- ` prefix |
| `Caption` | `*italic*` |
| `Table` | OTSL → restored **HTML table** |
| `Formula` | `$$ … $$` |
| `Picture` | `![Picture]()` placeholder (recognition skipped) |
| `Page-header` / `Page-footer` | Excluded from the body |

**JSON** — `{image, size, elements: [{label, box, text}]}`. `box` is in 0–999 normalized
coordinates, so it can be reused regardless of the original page size.

**OTSL** — the table serialization format ([tools/otsl.py](tools/otsl.py)). It roughly
halves tokens versus HTML and makes unclosed tags grammatically impossible.
`otsl_to_html()` converts back.

---

## 📁 Directory layout

```
Pierrot_VLM_OCR/
├── infer/
│   └── infer_pierrotocrvlm.py  # ★ single-page CLI — two-pass parsing, assembly, visualization
├── benchmark/
│   └── run_pages.py            # ★ batch runner — load once, mode/shard/diagnostic options
├── eval/
│   └── metrics_ocr.py          # layout output parser (parse_layout) + metrics (NED · TEDS · F1 · tau)
├── tools/
│   ├── pierrotocr_common.py    # single source of task prompts (= the task switch)
│   ├── otsl.py                 # OTSL ↔ HTML table converter
│   ├── hybrid_merge.py         # page ↔ c2f merging, column splitting, uncovered-band detection
│   ├── make_demo_viewer.py     # 3-panel replay viewer (page + RAW stream + reflow) as HTML
│   ├── compare_modes.py        # side-by-side comparison image for two modes
│   └── record_demo_gif.py      # viewer → demo GIF recording (needs playwright)
├── requirements.txt
├── LAB/pierrotocrvlm/          # one lab note — design, training, experiments, results, failures
├── docs/                       # architecture figure, LAB figures, banner
└── pierrot/
    └── models/pierrotocrvlm/   # algorithm package (inference paths only)
        ├── config.py           #   PierrotOCRConfig / Text / Vision (1:1 with HF config.json)
        ├── modeling/
        │   ├── vision.py       #     dynamic-resolution ViT + bilinear pos interpolation + patch merger + DeepStack
        │   ├── text.py         #     Qwen3 decoder (RMSNorm + M-RoPE + QK-Norm + GQA + SwiGLU, + KVCache)
        │   └── pierrotocrvlm.py#     merge (masked_scatter) + M-RoPE positions + generate (cycle / n-gram guards)
        ├── processor.py        #   dynamic-resolution patching (smart_resize) + ChatML prompt encoding
        ├── detection.py        #   detection/layout output parsing & visualization
        └── weights.py          #   checkpoint loading (config + safetensors/model.pt + sidecar)
```

---

## 🧪 LAB notes

Design, training method, experiments, and results are consolidated into **one document**:
[**LAB/pierrotocrvlm/pierrotocrvlm.md**](LAB/pierrotocrvlm/pierrotocrvlm.md) (written in Korean).

It does not record only what worked. **15 rejected hypotheses** and **13 measurement
defects** are in there too — those were the most expensive part of this project.

| Section | Content |
|---|---|
| 1 Overview · 2 Architecture | What changed from the MinerU2.5 skeleton · lineage · why the 4 mergers were the biggest risk |
| 3 Tasks · 4 Inference | Prompt = task switch · three detection conventions · OTSL · assembly +11.81p · band re-detection |
| 5 Training · 6 Data | Stage 0–3 curriculum · **blend by tokens, not records** · data lessons and 3 incidents |
| 7 Experiments · 8 Performance | v1 → v2 → v3 → three straight losses → teacher distillation → V3.5/3.6 · leaderboard comparison |
| 9 Rejected hypotheses · 10 Measurement defects · 11 Remaining bottlenecks | What not to retry · suspect the instrument first · stage-by-stage survival |

Where the document mentions `args/`, `training/`, or `tools/build_*`, it refers to the
**training repository** — those files are not in this distribution.

---

## 🚫 What is not here

| Missing | Where it lives |
|---|---|
| Training entrypoint `training/train_pierrotocrvlm.py` | Training repository |
| Hyperparameter single source `args/pierrotocrvlm.py` | ditto |
| Accelerate training engine `pierrot/core` (Trainer, scheduler, registry) | ditto |
| Dataset adapters, collate, augmentation (`dataset.py`, `pierrot/data`) | ditto |
| Data builders `tools/build_*_jsonl.py` (30-odd files) | ditto |
| Part-transplant assembly `weights.load_hybrid()` (the training starting point) | ditto |
| Val-set evaluation harness `eval/eval_pierrotocrvlm.py` | ditto |

The model code (`config`, `modeling`, `processor`, `weights`) is the **same
implementation** as in the training repository; only training-only paths were removed —
loss computation, activation checkpointing, optimizer parameter groups, and label
construction. Checkpoint compatibility is unchanged.

---

## 📚 Reference

Paper notes and reviews → [Pierrot-vision/Reading-Papers — VLM](https://github.com/Pierrot-vision/Reading-Papers#-vlm)

- **MinerU2.5** — the coarse-to-fine two-pass document parsing skeleton
- **Qwen3-VL** / **Qwen3** — vision tower and language decoder parts
- **OTSL** — table structure serialization
- **KDoc-OCRBench-V2** · **OmniDocBench** · **CC-OCR** — evaluation benchmarks

---

## 📄 License

This project (code and documentation) is licensed under **CC BY-NC-SA 4.0** — attribution, non-commercial, share-alike. See [LICENSE](LICENSE) for details. (Third-party datasets, models, and libraries follow their own licenses.)

> ⛔ **NonCommercial** — for non-commercial use only. Please contact us separately for commercial use.

---

## 📮 Contact

- Please reach out via [email](mailto:peternara@naver.com) or a [GitHub Issue](https://github.com/Pierrot-vision/Pierrot-VLM/issues). I will answer as best I can.
- Note that questions already answered on GitHub (README, code, docs) may not receive a reply.
