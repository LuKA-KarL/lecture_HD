"""저장된 MNIST CNN 모델을 테스트 데이터로 평가하는 스크립트"""
import os

from tensorflow import keras

from data_utils import load_mnist_data
from logger_utils import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn_model.keras")


def main():
    if not os.path.exists(MODEL_PATH):
        logger.error(
            "학습된 모델을 찾을 수 없습니다: %s (먼저 run.py를 실행하세요)", MODEL_PATH
        )
        return

    try:
        _, _, (x_test, y_test) = load_mnist_data()

        logger.info("모델 로딩: %s", MODEL_PATH)
        model = keras.models.load_model(MODEL_PATH)

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        logger.info(
            "테스트 결과 - loss: %.4f, accuracy: %.4f", test_loss, test_accuracy
        )
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

    except Exception:
        logger.exception("테스트 중 오류가 발생했습니다.")
        raise


if __name__ == "__main__":
    main()
