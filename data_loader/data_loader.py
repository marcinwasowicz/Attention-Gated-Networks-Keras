import os
import math
from typing import List, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

from .utils import get_files_from_dir, load_nii_image, load_dcm_image


class DataLoader(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size: int,
        input_image_paths: List[str],
        target_cache: Dict[Tuple[int, int], Tuple[str, np.ndarray]],
    ):
        self._batch_size = batch_size
        self._input_image_paths = input_image_paths
        self._target_cache = target_cache

    def __len__(self):
        return math.ceil(len(self._input_image_paths) / self._batch_size)

    def __getitem__(self, index):
        x_paths = self._input_image_paths[
            index * self._batch_size : (index + 1) * self._batch_size
        ]
        x = np.array([load_dcm_image(path) for _, _, path in x_paths])
        y = np.array(
            [self._target_cache[dir_id][:, :, img_id] for dir_id, img_id, _ in x_paths]
        )
        return x, y


class DataLoaderFactory:
    def __init__(
        self,
        patients_inputs_dir: str,
        patients_target_dir: str,
        input_file_ext: str,
        target_file_ext: str,
        batch_size: int = 32,
    ):
        self.batch_size = batch_size
        self._input_images_paths = shuffle(
            [
                (directory_index, image_index, image_file)
                for directory_index, patient_input in enumerate(
                    filter(
                        lambda x: os.path.isdir(x.path),
                        sorted(os.scandir(patients_inputs_dir), key=lambda x: x.path),
                    )
                )
                for image_index, image_file in enumerate(
                    sorted(get_files_from_dir(patient_input.path, input_file_ext))
                )
            ],
            random_state=42,
        )

        self._target_images_cache = {
            directory_index: load_nii_image(target_batch_path)
            for directory_index, target_batch_path in enumerate(
                sorted(get_files_from_dir(patients_target_dir, target_file_ext))
            )
        }

    def produce_loaders(
        self, test_size: float = None, val_size: float = None, batch_size: int = None
    ):
        if not batch_size:
            batch_size = 32

        if not test_size:
            return DataLoader(
                batch_size, self._input_images_paths, self._target_images_cache
            )

        train, test = train_test_split(self._input_images_paths, test_size=test_size)
        if not val_size:
            return DataLoader(batch_size, train, self._target_images_cache), DataLoader(
                batch_size, test, self._target_images_cache
            )

        train, val = train_test_split(train, test_size=val_size)
        return (
            DataLoader(batch_size, train, self._target_images_cache),
            DataLoader(batch_size, val, self._target_images_cache),
            DataLoader(batch_size, test, self._target_images_cache),
        )
