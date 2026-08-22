"""Pierrot-VLM-OCR: PierrotOCRVLM 문서 파싱 **추론** 패키지.

[Pierrot-VLM](https://github.com/Pierrot-vision/Pierrot-VLM) 에서 학습한 문서 파싱
모델(PierrotOCRVLM)을 돌리는 데 필요한 부분만 떼어낸 배포본이다. 학습 엔진
(Accelerate 루프)·데이터 빌더·데이터셋 어댑터는 들어 있지 않다 — 체크포인트를
읽어 생성하는 경로만 있다.

    from pierrot.models.pierrotocrvlm import load_pretrained
    model, processor = load_pretrained("outputs/pierrotocrvlm_v3/final", device="cuda")
"""

__all__ = ["models"]

__version__ = "0.1.0"
