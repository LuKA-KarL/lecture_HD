"""MNIST 필기체 인식 CNN 모델 학습 스크립트"""
import os

from tensorflow import keras

from data_utils import load_mnist_data
from model import build_model
from metrics_logger import EpochMetricsSaver
from logger_utils import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn_model.keras")
DEFAULT_EPOCHS = 50


def main():
    try:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()

        model = build_model()
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary(print_fn=logger.info)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
        metrics_saver = EpochMetricsSaver()

        logger.info("모델 학습 시작 - epochs: %d", DEFAULT_EPOCHS)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=DEFAULT_EPOCHS,
            batch_size=128,
            callbacks=[early_stopping, metrics_saver],
        )

        model.save(MODEL_PATH)
        logger.info("모델 저장 완료: %s", MODEL_PATH)

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        logger.info(
            "최종 테스트 결과 - loss: %.4f, accuracy: %.4f", test_loss, test_accuracy
        )

    except Exception:
        logger.exception("학습 중 오류가 발생했습니다.")
        raise


if __name__ == "__main__":
    main()
