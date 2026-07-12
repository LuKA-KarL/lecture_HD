"""에포크별 학습 지표를 optimizer/result/error/accuracy 폴더에 저장하는 콜백"""
import json
import os

from tensorflow import keras

from logger_utils import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(__file__)


class EpochMetricsSaver(keras.callbacks.Callback):
    """매 에포크 종료 시 optimizer/result/error/accuracy 값을 파일로 저장한다."""

    def __init__(self):
        super().__init__()
        self.dirs = {
            "optimizer": os.path.join(BASE_DIR, "optimizer"),
            "result": os.path.join(BASE_DIR, "result"),
            "error": os.path.join(BASE_DIR, "error"),
            "accuracy": os.path.join(BASE_DIR, "accuracy"),
        }
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1
        filename = f"epoch_{epoch_num:03d}.json"

        loss_data = {
            "epoch": epoch_num,
            "loss": logs.get("loss"),
            "val_loss": logs.get("val_loss"),
        }
        accuracy_data = {
            "epoch": epoch_num,
            "accuracy": logs.get("accuracy"),
            "val_accuracy": logs.get("val_accuracy"),
        }
        result_data = {**loss_data, **accuracy_data}
        optimizer_data = {
            "epoch": epoch_num,
            "learning_rate": float(
                keras.backend.get_value(self.model.optimizer.learning_rate)
            ),
        }

        self._write_json(os.path.join(self.dirs["error"], filename), loss_data)
        self._write_json(os.path.join(self.dirs["accuracy"], filename), accuracy_data)
        self._write_json(os.path.join(self.dirs["result"], filename), result_data)
        self._write_json(os.path.join(self.dirs["optimizer"], filename), optimizer_data)

        logger.info(
            "에포크 %d 저장 완료 - loss: %.4f, val_loss: %.4f, "
            "accuracy: %.4f, val_accuracy: %.4f",
            epoch_num,
            loss_data["loss"] or 0.0,
            loss_data["val_loss"] or 0.0,
            accuracy_data["accuracy"] or 0.0,
            accuracy_data["val_accuracy"] or 0.0,
        )

    @staticmethod
    def _write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
