# 필기체 인식 프로젝트 (MNIST Handwriting Recognition)

MNIST dataset을 사용한 필기체 숫자 인식 CNN 모델. Keras 기반 4-layer CNN으로 학습하며, 에포크마다 학습 지표를 기록한다.

## 프로젝트 구조

```
.
├── requirements.txt
├── CLAUDE.md
└── main/               # 메인 코드와 실행 파일
    ├── data_utils.py   # MNIST 로드, 32x32 패딩, 정규화, train/val/test 분리
    ├── model.py        # 4-layer CNN 정의
    ├── metrics_logger.py  # 에포크별 지표 저장 콜백
    ├── logger_utils.py # 로거 유틸리티
    ├── run.py          # 학습 실행 스크립트
    ├── test.py         # 저장된 모델 평가 스크립트
    ├── optimizer/      # 에포크별 optimizer(learning rate) 값 저장
    ├── result/         # 에포크별 accuracy + loss 저장
    ├── error/          # 에포크별 loss 저장
    └── accuracy/       # 에포크별 accuracy 저장
```

## 설치 및 실행

```bash
# 가상환경 생성 (Python 3.12 고정 — TensorFlow 호환)
uv venv --python 3.12 .venv

# 의존성 설치
uv pip install --python .venv/bin/python -r requirements.txt

# 학습 실행 (main/mnist_cnn_model.keras 생성)
cd main && ../.venv/bin/python3 run.py

# 저장된 모델 평가
cd main && ../.venv/bin/python3 test.py
```

## 모델 사양

- 입력: 32×32×1 (MNIST 원본 28×28 이미지를 zero-padding으로 확장)
- Conv2D 4개 레이어 (32 → 64 → 128 → 128 채널), 각 레이어에 BatchNormalization 적용
- Dropout(0.5) + EarlyStopping(`monitor="val_loss", patience=5`)으로 과적합 방지
- Optimizer: Adam, Loss: categorical_crossentropy
- default epochs: 50 (EarlyStopping으로 조기 종료 가능)

## 학습 결과 (예시)

50 epoch 설정으로 학습 시 12 epoch에서 EarlyStopping이 발동했고, 최종 테스트 결과는 다음과 같다.

| 지표 | 값 |
|---|---|
| Test Loss | 0.0241 |
| Test Accuracy | 99.30% |

## 기술 스택

- Python 3.12
- uv (패키지/환경 관리)
- TensorFlow / Keras
- NumPy
