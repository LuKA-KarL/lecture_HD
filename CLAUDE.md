# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 필기체 인식 프로젝트 (MNIST Handwriting Recognition)

## Project Overview
MNIST dataset을 사용해서 필기체 인식 프로그램을 개발하는 프로젝트.
`main/` 폴더에 4-layer CNN(Keras) 학습/테스트 코드가 구현되어 있다.

## Commands
```bash
# 가상환경 생성 (시스템 기본 python3가 최신 버전이라 TensorFlow 미지원일 수 있으므로 3.12 고정)
uv venv --python 3.12 .venv

# 의존성 설치
uv pip install --python .venv/bin/python -r requirements.txt

# 학습 실행 (main/ 디렉터리에서 실행, mnist_cnn_model.keras 생성)
cd main && ../.venv/bin/python3 run.py

# 테스트 실행 (main/ 디렉터리에서 실행, 학습된 모델 필요)
cd main && ../.venv/bin/python3 test.py
```

## Tech Stack
- Python 3.12 (`.venv`로 고정 — 시스템 python3가 3.13+이면 TensorFlow가 설치되지 않으므로 반드시 `uv venv --python 3.12`로 venv 생성)
- uv (패키지/환경 관리)
- Keras (tensorflow 패키지에 포함, `tensorflow.keras`)
- MNIST dataset — **반드시 이미지 픽셀 사이즈 32×32 이상으로 사용**
  (MNIST 원본은 28×28이므로 `data_utils.py`에서 zero-padding으로 32×32로 확장)
- 의존성은 requirements.txt로 관리

## Architecture

### 모델 요구사항
- Keras `Sequential` 기반 CNN (`main/model.py`)
- **Conv2D 4개 레이어** (32→64→128→128 채널, BatchNorm + MaxPooling 포함) + Dense 분류 헤드
- EarlyStopping(`monitor="val_loss", patience=5, restore_best_weights=True`)과 Dropout(0.5)으로 과적합 방지
- default epochs: 50 (`main/run.py`의 `DEFAULT_EPOCHS`)
- 학습 데이터의 10%를 검증셋으로 분리 (`data_utils.load_mnist_data`의 `val_split`)
- 학습 완료 후 모델은 `main/mnist_cnn_model.keras`로 저장되며, `test.py`가 이 파일을 로드해 평가

### 모듈 구성 (`main/`)
- `data_utils.py` — MNIST 로드, 32×32 패딩, 정규화, train/val/test 분리
- `model.py` — 4-layer CNN 정의 (`build_model`)
- `metrics_logger.py` — `EpochMetricsSaver` 콜백: 매 에포크 종료 시 4개 폴더에 JSON 저장
- `logger_utils.py` — 모듈별 콘솔 로거 생성 (`get_logger`)
- `run.py` — 학습 진입점
- `test.py` — 저장된 모델 평가 진입점 (모델 파일 없으면 logger.error 후 종료)

### 폴더 구조
```
main/           # 메인 코드와 실행 파일이 저장되는 폴더
├── optimizer/  # 매 에포크의 learning rate 등 optimizer 상태 저장 (epoch_NNN.json)
├── result/     # 각 에포크마다 accuracy와 loss를 함께 저장 (epoch_NNN.json)
├── error/      # 각 에포크마다 loss(loss, val_loss) 저장 (epoch_NNN.json)
└── accuracy/   # 각 에포크마다 accuracy(accuracy, val_accuracy) 저장 (epoch_NNN.json)
```
네 폴더 모두 `EpochMetricsSaver` 콜백(`main/metrics_logger.py`)이 `on_epoch_end`에서 채운다.

## Code Style
- 주석: 한글로 작성
- 에러 처리: logger를 사용해서 기록 (logging 모듈 사용)
- 함수명: snake_case

## Development Notes
- 제한사항: 과적합(overfitting)이 발생하지 않도록 코딩할 것
