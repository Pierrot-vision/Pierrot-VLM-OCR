"""PierrotOCRVLM 알고리즘 패키지 — 문서 파싱(OCR) 전용 ≈1.0B VLM (추론 전용 배포본).

기본 알고리즘 = MinerU2.5(단일 체크포인트, 프롬프트 전환식 coarse-to-fine 2단계).
부품 = Qwen3-VL-2B ViT(+DeepStack) + Qwen3-0.6B 디코더(M-RoPE), 하이브리드 이식.

학습 배포본과 달리 레지스트리 등록(spec)·데이터셋 어댑터가 없다. 체크포인트를 읽는
load_pretrained() 와 생성 경로만 노출한다.
"""

from .config import PierrotOCRConfig, PierrotOCRTextConfig, PierrotOCRVisionConfig
from .modeling.pierrotocrvlm import PierrotOCRForConditionalGeneration
from .processor import PierrotOCRProcessor
from .weights import build_model_from_config, load_pretrained

__all__ = [
    "PierrotOCRConfig",
    "PierrotOCRTextConfig",
    "PierrotOCRVisionConfig",
    "PierrotOCRForConditionalGeneration",
    "PierrotOCRProcessor",
    "load_pretrained",
    "build_model_from_config",
]
