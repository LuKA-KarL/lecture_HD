"""MNIST 데이터 로딩 및 전처리 유틸리티"""
import numpy as np
from tensorflow import keras

from logger_utils import get_logger

logger = get_logger(__name__)

IMG_SIZE = 32  # 필수 요구사항: 이미지 픽셀 사이즈 32x32 이상 사용
PAD = (IMG_SIZE - 28) // 2  # 28x28 원본 이미지를 32x32로 패딩


def _pad_and_normalize(images: np.ndarray) -> np.ndarray:
    """28x28 이미지를 32x32로 패딩하고 [0, 1] 범위로 정규화한다."""
    padded = np.pad(
        images,
        ((0, 0), (PAD, PAD), (PAD, PAD)),
        mode="constant",
        constant_values=0,
    )
    normalized = padded.astype("float32") / 255.0
    return normalized[..., np.newaxis]  # 채널 차원 추가


def load_mnist_data(val_split: float = 0.1):
    """MNIST 데이터셋을 로드하고 학습/검증/테스트 세트로 전처리하여 반환한다."""
    logger.info("MNIST 데이터셋 로딩 시작")
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train_full = _pad_and_normalize(x_train_full)
    x_test = _pad_and_normalize(x_test)

    y_train_full = keras.utils.to_categorical(y_train_full, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    val_count = int(len(x_train_full) * val_split)
    x_val, y_val = x_train_full[:val_count], y_train_full[:val_count]
    x_train, y_train = x_train_full[val_count:], y_train_full[val_count:]

    logger.info(
        "데이터 로딩 완료 - train: %d, val: %d, test: %d",
        len(x_train),
        len(x_val),
        len(x_test),
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
