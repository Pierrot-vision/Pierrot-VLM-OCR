"""PierrotOCRVLM 태스크 프롬프트 (단일 소스).

이 모델에는 태스크별 헤드가 하나도 없다 — **프롬프트 문자열이 곧 태스크 스위치**다
(MinerU2.5 방식). 레이아웃 검출·텍스트/표/수식 인식·페이지 통읽기가 모두 같은
체크포인트에서 프롬프트만 바꿔 나온다.

그래서 상수를 여기 한 곳에서만 정의하고 추론·병합·평가가 전부 이것을 참조한다.
학습 정답(JSONL 의 prefix)에 쓴 문자열과 한 글자라도 어긋나면 모델이 **다른
태스크로 알아듣는다** — 예를 들어 "Layout Detection (fine):" 과
"Layout Detection (coarse):" 는 라벨 규약 자체가 다른 별개의 태스크다.

아래 상수 블록은 학습 저장소(Pierrot-VLM)의 같은 파일과 **줄 단위로 동일**하다.
학습 쪽 데이터 빌더 유틸(split_train_val · write_jsonl 등)은 추론에 쓰이지 않아
이 배포본에서 뺐다.
"""



# ── 태스크 프롬프트 (단일 소스) ──
PROMPT_LAYOUT  = "Layout Detection:"
# ★ 레이아웃 라벨 규약 분리(v2): 두 gold 데이터셋의 라벨 규약이 실측으로 달랐다 —
#   ccpdf 는 페이지당 중앙값 18개로 잘게 쪼개고 Caption 9.9%·List-item 13.9% 를 쓰는데,
#   DocLayNet 은 11개로 묶고 Caption 1.7%·Text 51.6% 다. 같은 시각 요소에 상충하는
#   정답을 주니 모델이 Caption↔Text·Section-header↔Text 를 혼동했다(오분류 13.1%).
#   프롬프트로 "어느 규약으로 라벨링할지"를 명시해 조건부로 학습시킨다.
#   추론 시에는 우리 표준(fine)을 쓰고, coarse 는 DocLayNet 물량을 살리는 용도다.
PROMPT_LAYOUT_FINE   = "Layout Detection (fine):"    # ccpdf 규약 — 세밀 분할, 우리 표준
PROMPT_LAYOUT_COARSE = "Layout Detection (coarse):"  # DocLayNet 규약 — 묶음 분할
# 야외 텍스트(간판·표지판·상품·책표지) 규약 — 클래스는 Text 하나, 단위는 **어절**이다.
#   문서 레이아웃(문단 덩어리)과 규약이 전혀 다르므로 같은 프롬프트를 쓰면 서로를 오염시킨다.
#   fine/coarse 와 같은 이유로 세 번째 규약을 둔다. 이게 있어야 2단계 추론
#   (검출 → crop 인식)이 문서뿐 아니라 **일반 사진**에서도 동작한다.
PROMPT_LAYOUT_WILD   = "Layout Detection (wild):"    # 야외 텍스트 — 어절 박스
PROMPT_TEXT    = "Text Recognition:"
PROMPT_FORMULA = "Formula Recognition:"
PROMPT_TABLE   = "Table Recognition:"
# 페이지 통읽기(위키 렌더류): 원본 GT 프롬프트가 이미 다국어 지시문이므로
# 태스크 마커 + 원본 지시문을 이어 쓴다.
PROMPT_PAGE    = "Page Recognition:"
