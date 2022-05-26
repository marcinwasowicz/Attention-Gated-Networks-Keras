import json
import os
import pickle

from data_loader import DataLoaderFactory
from models import Unet3DFactory
from tensorflow.keras.callbacks import ModelCheckpoint


class CheckpointManager(ModelCheckpoint):
    def __init__(
        self, path_prefix, validation_loader, prediction_freq, *args, **kwargs
    ):
        super().__init__(filepath=os.path.join(path_prefix, "model"), *args, **kwargs)
        self.path = path_prefix
        self.prediction_freq = prediction_freq
        self.validation_loader = validation_loader
        self.validation_loader._batch_size = 1
        self._metrics = {}

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        for k, v in logs.items():
            if k not in self._metrics:
                self._metrics[k] = []
            self._metrics[k].append(v)

        with open(os.path.join(self.path, "metrics.json"), "w+") as metrics_file:
            json.dump(self._metrics, metrics_file)

        if epoch % self.prediction_freq == 0:
            val_data_gen = (
                self.validation_loader[i] for i in range(len(self.validation_loader))
            )
            val_predictions = [
                (self.model.predict_on_batch(x), y) for x, y in val_data_gen
            ]
            with open(
                os.path.join(self.path, f"predictions_{epoch}.pkl"), "wb+"
            ) as predictions_file:
                pickle.dump(val_predictions, predictions_file)


if __name__ == "__main__":
    with open("./config/unet_3D_config.json", "r") as f:
        config = json.load(f)

    train_loader, test_loader = DataLoaderFactory(**config).produce_loaders(
        test_size=0.2
    )
    unet = Unet3DFactory(**config).produce_unet()
    checkpoint_manager = CheckpointManager(**config, validation_loader=test_loader)
    unet.fit(
        train_loader,
        validation_data=test_loader,
        epochs=config["epochs"],
        callbacks=[checkpoint_manager],
    )
