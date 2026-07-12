"""4-layer CNN 기반 MNIST 분류 모델 정의"""
from tensorflow import keras
from tensorflow.keras import layers

from logger_utils import get_logger

logger = get_logger(__name__)

INPUT_SHAPE = (32, 32, 1)
NUM_CLASSES = 10


def build_model() -> keras.Model:
    """4개의 합성곱 레이어로 구성된 CNN 모델을 생성한다."""
    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),
            # 1번째 conv layer
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # 2번째 conv layer
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # 3번째 conv layer
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            # 4번째 conv layer
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # 분류기 (Dropout으로 과적합 방지)
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="mnist_cnn_4layer",
    )
    logger.info("모델 생성 완료 - 4-layer CNN")
    return model
