import numpy as np
from scipy import ndimage
import tensorflow as tf


class Preprocessing:
    _MIN_TISSUE_INTENSITY = -1000
    _MAX_TISSUE_INTENSITY = 400

    def __init__(self, resize_val, n_classes):
        self._resize_val = resize_val
        self._n_classes = n_classes
        self._preprocessing_pipelines = {
            "input": [
                self._clip_tissue_intensity,
                self._normalize_tissue_intensity,
                self._resize_image,
                self._add_dim,
            ],
            "target": [self._resize_image, self._add_dim, self._one_hot_encode],
        }

    def _clip_tissue_intensity(self, image: np.ndarray) -> np.ndarray:
        return tf.clip_by_value(
            image, self._MIN_TISSUE_INTENSITY, self._MAX_TISSUE_INTENSITY
        )

    def _normalize_tissue_intensity(self, image: np.ndarray) -> np.ndarray:
        return tf.math.divide(
            image, (self._MAX_TISSUE_INTENSITY - self._MIN_TISSUE_INTENSITY)
        )

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        width_factor = self._resize_val[0] / image.shape[0]
        height_factor = self._resize_val[1] / image.shape[1]
        depth_factor = self._resize_val[2] / image.shape[2]
        image = ndimage.zoom(image, (width_factor, height_factor, depth_factor))
        return image

    def _add_dim(self, image: np.ndarray) -> np.ndarray:
        return tf.expand_dims(image, axis=3)

    def _one_hot_encode(self, image: np.ndarray) -> np.ndarray:
        return tf.keras.utils.to_categorical(image, self._n_classes)

    def __call__(self, image: np.ndarray, image_type: str) -> np.ndarray:
        for fun in self._preprocessing_pipelines[image_type]:
            image = fun(image)
        return image
