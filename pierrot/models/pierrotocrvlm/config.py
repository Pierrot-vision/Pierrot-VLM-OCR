"""PierrotOCRVLM 모델 설정.

PierrotOCRVLM 은 문서 파싱(OCR) 전용 ≈1.0B VLM 이다. 기본 알고리즘은 MinerU2.5
(단일 체크포인트가 프롬프트 전환만으로 레이아웃 검출과 영역 인식을 모두 수행하는
coarse-to-fine 2단계)이고, 부품은 Qwen3 세대로 교체했다:

    비전 타워   : Qwen3-VL-2B 의 동적 해상도 ViT (+DeepStack)   ← 사전학습 가중치 이식
    머저 4개    : 본 머저 1 + DeepStack 머저 3 (출력 1024)      ← 신규(랜덤/zero-init)
    언어 디코더 : Qwen3-0.6B (M-RoPE 로 교체)                    ← 사전학습 가중치 이식

즉 "modular initialization": 비전/언어는 검증된 사전학습 부품에서 오고, 두 부품을
잇는 머저와 문서 파싱 능력은 처음부터 학습한다. 랜덤 초기화되는 경계가 머저 4개
전체라는 점이 이 모델의 최초 학습 리스크이며, 로더(weights.py)가 DeepStack 머저의
출력층을 zero-init 해 주입을 no-op 에서 시작시킨다.

설정 원칙:
  - 필드 이름은 HF 계열 config.json 키와 맞춘다. 비전은 Qwen3-VL 체크포인트의
    vision_config 에서, 언어는 Qwen3-0.6B 체크포인트의 config 에서 그대로 읽어
    조립할 수 있다(부품 이식 조립은 학습 저장소 Pierrot-VLM 쪽 경로다).
  - 아래 dataclass 기본값 = "Qwen3-VL-2B ViT + Qwen3-0.6B" 하이브리드 그 자체다.
    체크포인트 없이 기본값만으로도 동일 구조가 만들어진다.

구조 요약(왜 이 조합인가):
  - 동적 해상도: 문서를 정사각 타일로 자르지 않고 원본 종횡비 그대로
    32(=patch 16 × merge 2) 배수 격자로 본다. 레이아웃 패스의 해상도 예산은
    max_pixels 로 조절한다(MinerU 의 1036 은 28 배수 그리드라 여기선 쓰지 않는다).
  - DeepStack: 비전 중간층(5/11/17) 특징을 디코더 앞쪽 레이어에 재주입한다.
    작은 글자 인식에 유리한 멀티레벨 피처 경로라 OCR 용도로 유지한다.
  - M-RoPE(interleaved): Qwen3-0.6B 의 head_dim 이 128 로 Qwen3-VL 과 같아
    mrope_section [24,20,20](합=64=head_dim/2)을 그대로 쓸 수 있다.
    텍스트 전용 입력(세 축 동일 위치)에서는 1D RoPE 와 수치적으로 일치해야 하며,
    학습 저장소의 tests/test_pierrotocrvlm.py 가 이를 회귀로 검증한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PierrotOCRVisionConfig:
    """비전 타워 설정 — Qwen3-VL-2B ViT 와 동일 구조 (out_hidden_size 만 다르다)."""

    hidden_size: int = 1024
    intermediate_size: int = 4096
    depth: int = 24                        # 비전 블록 수
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16                   # 공간 패치 한 변
    temporal_patch_size: int = 2           # 시간 패치(정지 이미지는 같은 프레임 2회 복제)
    spatial_merge_size: int = 2            # 패치 머저 압축 배율 m (이미지 토큰 = 패치수/m²)
    num_position_embeddings: int = 2304    # 학습형 위치 임베딩 격자 = 48×48
    # ★ Qwen3-VL-2B 원본은 2048(=그쪽 언어 hidden). 우리는 Qwen3-0.6B(1024)와 결합하므로
    #   1024 다. 이 차이 때문에 머저 4개(본 머저+DeepStack 머저)는 사전학습 가중치를
    #   재사용할 수 없고 전부 새로 학습한다 — 이 모델의 핵심 랜덤 초기화 경계.
    out_hidden_size: int = 1024
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    # DeepStack: 이 인덱스의 비전 블록 출력을 별도 머저로 뽑아 디코더 앞쪽 레이어에 주입.
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [5, 11, 17])

    # ------------------------------------------------------------------ #
    # 어텐션 헤드 하나의 차원 = hidden_size // num_heads.
    # ------------------------------------------------------------------ #
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    # ------------------------------------------------------------------ #
    # 학습형 위치 임베딩 격자의 한 변(= √num_position_embeddings, 공식 48).
    # 실제 이미지 격자(h, w)에는 이 격자를 bilinear 보간해 맞춘다.
    # ------------------------------------------------------------------ #
    @property
    def num_grid_per_side(self) -> int:
        return int(self.num_position_embeddings ** 0.5)

    # ------------------------------------------------------------------ #
    # 패치 하나를 펼친 벡터의 길이 = C × T × p × p (프로세서 출력 마지막 차원).
    # ------------------------------------------------------------------ #
    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size ** 2


@dataclass
class PierrotOCRTextConfig:
    """언어 디코더 설정 — Qwen3-0.6B 와 동일 구조 + M-RoPE 교체.

    Qwen3-0.6B 는 1D RoPE 텍스트 모델이지만, RoPE 에는 학습 파라미터가 없으므로
    가중치를 그대로 둔 채 위치 인코딩만 M-RoPE(3축)로 바꿔 끼운다. 텍스트 구간은
    세 축이 같은 위치를 공유하므로 원본과 수치적으로 동일하게 동작해야 하고
    (테스트로 검증), 이미지 구간에서만 격자 좌표가 갈라진다.
    """

    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8           # GQA (Q 16 : KV 8)
    head_dim: Optional[int] = 128          # Qwen3 는 hidden/heads 와 무관하게 128 고정
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True       # Qwen3-0.6B 는 lm_head 를 임베딩과 공유
    pad_token_id: Optional[int] = None
    # ── M-RoPE ──
    # head_dim/2(=64)개 주파수를 시간/높이/너비 세 축에 [24,20,20]으로 배분한다.
    # Qwen3-0.6B 의 head_dim(128)이 Qwen3-VL 과 같아 배분을 그대로 가져온다.
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    # True 면 [TTT...HHH...WWW] 청크가 아니라 [THWTHW...] 교차 배치(Qwen3-VL 방식).
    mrope_interleaved: bool = True

    # ------------------------------------------------------------------ #
    # head_dim 미지정 시 hidden_size / num_attention_heads 로 채운다.
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    # ------------------------------------------------------------------ #
    # GQA 그룹 수 = Q 헤드 수 / KV 헤드 수 (KV 헤드당 공유하는 Q 헤드 수).
    # ------------------------------------------------------------------ #
    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass
class PierrotOCRConfig:
    """최상위 PierrotOCRVLM 설정 (비전+언어를 묶고 특수 토큰을 배선)."""

    text_config: PierrotOCRTextConfig = field(default_factory=PierrotOCRTextConfig)
    vision_config: PierrotOCRVisionConfig = field(default_factory=PierrotOCRVisionConfig)

    # Qwen3 계열 토크나이저(vocab 151,936)의 비전 특수 토큰 id 들.
    # Qwen3-0.6B 토크나이저에도 예약돼 있어 텍스트 모델 출신 디코더와 충돌하지 않는다.
    image_token_id: int = 151655           # <|image_pad|> (이미지 placeholder)
    video_token_id: int = 151656           # <|video_pad|> (이미지 전용 구현 — 예약)
    vision_start_token_id: int = 151652    # <|vision_start|>
    vision_end_token_id: int = 151653      # <|vision_end|>
    ignore_index: int = -100
    vocab_size: int = 151936               # = text_config.vocab_size
    tie_word_embeddings: bool = True

    # ------------------------------------------------------------------ #
    # 하위 config 를 정돈하고 교차 배선을 맞춘다.
    #   - dict 로 들어온 text/vision config 를 dataclass 로 승격(config.json 대응)
    #   - 최상위 vocab/tie 를 언어 config 와 동기화
    #   - 머저 출력(out_hidden_size)과 언어 hidden 이 다르면 즉시 오류
    #     (다르면 이미지 임베딩을 텍스트 시퀀스에 끼울 수 없다)
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if isinstance(self.text_config, dict):
            self.text_config = PierrotOCRTextConfig(**_filter(PierrotOCRTextConfig, self.text_config))
        if isinstance(self.vision_config, dict):
            self.vision_config = PierrotOCRVisionConfig(**_filter(PierrotOCRVisionConfig, self.vision_config))

        self.vocab_size          = self.text_config.vocab_size
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"vision_config.out_hidden_size({self.vision_config.out_hidden_size}) 와 "
                f"text_config.hidden_size({self.text_config.hidden_size}) 가 달라 이미지 병합이 불가능합니다."
            )

    # ------------------------------------------------------------------ #
    # 격자 (grid_t, grid_h, grid_w) 하나가 만들어 내는 이미지 토큰 수.
    #   = t·h·w / m²  (머저가 m×m 이웃 패치를 하나로 합치므로)
    # 프로세서(placeholder 개수)와 모델(병합 검증)이 같은 식을 참조한다.
    # ------------------------------------------------------------------ #
    def image_tokens_for_grid(self, grid_t: int, grid_h: int, grid_w: int) -> int:
        return grid_t * grid_h * grid_w // (self.vision_config.spatial_merge_size ** 2)


# ------------------------------------------------------------------ #
# dataclass 필드에 존재하는 키만 남긴다 (HF config.json 의 여분 키 무시).
# ------------------------------------------------------------------ #
def _filter(cls, d: dict) -> dict:
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in d.items() if k in valid}
