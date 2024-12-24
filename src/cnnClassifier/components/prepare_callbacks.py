import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config:PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,  # Use the updated .keras filepath
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        )

    @property
    def _create_tb_callbacks(self):
        log_dir = self.config.tensorboard_root_log_dir
        return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]
