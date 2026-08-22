"""PierrotOCRVLM 가중치 로딩 (추론 전용).

학습 산출물 체크포인트 하나를 읽어 (model, processor) 를 만든다.

    load_pretrained()        : 우리 학습 산출물(engine.save_pretrained 결과) 로드.
                               config.json + safetensors(또는 model.pt) + 토크나이저
                               + 전처리 sidecar(pierrotocrvlm_preprocessor.json).
    build_model_from_config(): 가중치 없이 구조만(랜덤) — 구조 점검용.

체크포인트의 구조 규약(학습 배포본에서 조립한 하이브리드 그대로):

    ┌─ 비전 타워  model.visual.*           (Qwen3-VL-2B ViT 계보 + DeepStack)
    ├─ 머저 4개  model.visual.merger.* / deepstack_merger_list.*
    └─ 언어 디코더 model.language_model.*  (Qwen3-0.6B 계보 + M-RoPE, lm_head 는 tie)

로드 검증은 엄격하다 — tie 로 채워지는 lm_head 외에 누락/여분 키가 있으면 조용한
랜덤 초기화 대신 즉시 예외를 낸다(잘못된 config/체크포인트 조합 조기 감지).

★ 하이브리드 최초 조립(공개 체크포인트 두 개에서 부품 이식)은 **학습 경로**라
  이 배포본에 없다 — 학습 저장소(Pierrot-VLM)의 weights.load_hybrid 를 쓴다.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional, Tuple

import torch

from .config import PierrotOCRConfig, _filter
from .modeling.pierrotocrvlm import PierrotOCRForConditionalGeneration
from .processor import PierrotOCRProcessor


# ------------------------------------------------------------------ #
# 로컬 경로면 그대로, Hub id 면 로컬 경로를 반환한다.
# ★ 오프라인 우선: 캐시에 스냅샷이 있으면 **네트워크에 전혀 접속하지 않고**
#   (local_files_only) 그 경로를 쓴다. 캐시가 없을 때만 실제 다운로드를 시도하고,
#   그마저 실패하면(네트워크 차단 등) 원인이 보이는 예외를 그대로 전파한다.
#   덕분에 오프라인·폐쇄망 추론이 HF HEAD 요청 재시도에 발목 잡히지 않는다.
# ------------------------------------------------------------------ #
def resolve_model_dir(model_id_or_path: str, revision: Optional[str] = None,
                      cache_dir: Optional[str] = None) -> str:
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    from huggingface_hub import snapshot_download

    patterns = ["*.safetensors", "*.json", "*.txt", "tokenizer*", "vocab.json", "merges.txt"]
    try:
        return snapshot_download(
            repo_id=model_id_or_path, revision=revision, cache_dir=cache_dir,
            allow_patterns=patterns, local_files_only=True,
        )
    except Exception:
        return snapshot_download(
            repo_id=model_id_or_path, revision=revision, cache_dir=cache_dir,
            allow_patterns=patterns,
        )


# ------------------------------------------------------------------ #
# 디렉토리의 가중치를 하나의 state_dict 로 읽는다.
#   - *.safetensors 여러 샤드를 합침(공개 체크포인트)
#   - 없으면 model.pt 로드(우리 학습 산출물, engine.save_pretrained)
# key_filter 를 주면 **필터를 통과한 키만 읽는다** — 체크포인트의 일부만 필요할 때
# CPU 메모리 피크를 줄인다(safetensors 는 키 단위 lazy read 라 걸러진 텐서는 아예
# 메모리에 올라오지 않는다).
# ------------------------------------------------------------------ #
def _load_state_dict(model_dir: str, key_filter=None) -> dict:
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        pt = os.path.join(model_dir, "model.pt")
        if os.path.exists(pt):
            state = torch.load(pt, map_location="cpu")
            return {k: v for k, v in state.items() if key_filter(k)} if key_filter else state
        raise FileNotFoundError(f"{model_dir} 에 *.safetensors 도 model.pt 도 없습니다.")
    tensors = {}
    from safetensors import safe_open

    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key_filter is None or key_filter(key):
                    tensors[key] = f.get_tensor(key)
    return tensors


# ------------------------------------------------------------------ #
# 학습 산출물의 config.json 을 읽는다(학습 엔진이 asdict 로 전 필드를 기록해 둔다).
# 하위 dict(text_config/vision_config)는 PierrotOCRConfig.__post_init__ 이 승격한다.
# ------------------------------------------------------------------ #
def config_from_json(model_dir: str) -> PierrotOCRConfig:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    top = _filter(PierrotOCRConfig, raw)
    return PierrotOCRConfig(**top)


# ------------------------------------------------------------------ #
# 체크포인트의 토크나이저를 로드해 PierrotOCRProcessor 를 만든다(우측 패딩).
# 픽셀 예산(min_pixels/max_pixels)·system_prompt 우선순위:
#   ① 호출자 명시(proc_kwargs 값 not None)
#   ② 우리 산출물 sidecar(pierrotocrvlm_preprocessor.json)
#   ③ 호출자 폴백(fallback — 애플리케이션 기본값)
# 학습 때 쓴 픽셀 예산은 sidecar 로 동봉되므로, 산출물 로드는 ②에서 자동 복원된다.
# ------------------------------------------------------------------ #
def build_processor(model_dir: str, config: PierrotOCRConfig,
                    fallback: Optional[dict] = None, **proc_kwargs) -> PierrotOCRProcessor:
    from transformers import AutoTokenizer

    sidecar = _read_sidecar(model_dir)
    _validate_sidecar_structure(sidecar, config)
    for key in ("min_pixels", "max_pixels", "layout_max_pixels", "system_prompt"):
        if proc_kwargs.get(key) is None and sidecar.get(key) is not None:
            proc_kwargs[key] = sidecar[key]
    if fallback:
        for key, val in fallback.items():
            if proc_kwargs.get(key) is None and val is not None:
                proc_kwargs[key] = val

    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    return PierrotOCRProcessor(tokenizer, config, **proc_kwargs)


# ------------------------------------------------------------------ #
# sidecar 의 구조 값(patch_size/merge_size)이 체크포인트 config 와 일치하는지 검증.
# 값이 없으면 통과, 다르면 잘못된 전처리 조합이므로 예외.
# ------------------------------------------------------------------ #
def _validate_sidecar_structure(sidecar: dict, config: PierrotOCRConfig) -> None:
    if not sidecar:
        return
    checks = (("patch_size", config.vision_config.patch_size),
              ("merge_size", config.vision_config.spatial_merge_size))
    for key, expected in checks:
        val = sidecar.get(key)
        if val is not None and val != expected:
            raise ValueError(
                f"[pierrotocrvlm:weights] sidecar {key}({val}) 가 체크포인트 config "
                f"{key}({expected}) 와 다릅니다 — 잘못된 모델/전처리 조합입니다."
            )


# ------------------------------------------------------------------ #
# 우리 학습 산출물의 pierrotocrvlm_preprocessor.json 을 읽는다(없으면 {}).
# 존재하는데 손상됐으면 조용히 넘기지 않고 예외를 낸다(전처리 불일치 조기 감지).
# ------------------------------------------------------------------ #
def _read_sidecar(model_dir: str) -> dict:
    path = os.path.join(model_dir, "pierrotocrvlm_preprocessor.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ #
# 우리 학습 산출물(또는 로컬 디렉토리) 체크포인트에서 (model, processor) 를 로드한다.
# 흐름: 경로 해석 → config.json → 프로세서(sidecar 픽셀 예산 복원) →
#       모델 생성 → state_dict 로드 → (tie) → dtype/device 이동.
# ------------------------------------------------------------------ #
def load_pretrained(
    model_id_or_path: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    fallback_proc_kwargs: Optional[dict] = None,
    **proc_kwargs,
) -> Tuple[PierrotOCRForConditionalGeneration, PierrotOCRProcessor]:
    model_dir = resolve_model_dir(model_id_or_path, revision=revision, cache_dir=cache_dir)
    config    = config_from_json(model_dir)
    processor = build_processor(model_dir, config, fallback=fallback_proc_kwargs, **proc_kwargs)
    config.image_token_id = processor.image_token_id

    model               = PierrotOCRForConditionalGeneration(config)
    state               = _load_state_dict(model_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if "lm_head.weight" in missing and config.tie_word_embeddings:
        model.tie_weights()
    _verify_load(model, state, missing, unexpected)

    if dtype is not None:
        model = model.to(dtype)
    return model.to(device), processor


# ------------------------------------------------------------------ #
# 가중치 없이 config 만으로 모델 생성(완전 랜덤 초기화, 구조 실험용).
# ------------------------------------------------------------------ #
def build_model_from_config(config: PierrotOCRConfig) -> PierrotOCRForConditionalGeneration:
    model = PierrotOCRForConditionalGeneration(config)
    if config.tie_word_embeddings:
        model.tie_weights()
    return model


# ------------------------------------------------------------------ #
# 산출물 로드 검증(엄격): tie 로 채워지는 lm_head 외의 누락/여분 키는 예외.
# ------------------------------------------------------------------ #
def _verify_load(model, state, missing, unexpected) -> None:
    allowed_missing = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    total  = len(model.state_dict())
    loaded = total - len(missing)
    print(f"[pierrotocrvlm:weights] 체크포인트에서 로드된 텐서 키: {loaded}/{total} "
          f"(체크포인트 제공 {len(state)}개)")

    hard_missing = [m for m in missing if m not in allowed_missing]
    if hard_missing:
        raise RuntimeError(
            f"[pierrotocrvlm:weights] 누락 키 {len(hard_missing)}개(해당 레이어가 랜덤 초기화됨): "
            f"{hard_missing[:8]} ... config 와 체크포인트가 일치하는지 확인하세요."
        )
    if unexpected:
        raise RuntimeError(
            f"[pierrotocrvlm:weights] 예상 밖 키 {len(unexpected)}개: {unexpected[:8]} ... "
            f"체크포인트 포맷/디렉토리를 확인하세요."
        )
