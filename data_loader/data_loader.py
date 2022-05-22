import os
import math
from typing import List, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

from .preprocessing import Preprocessing
from .utils import get_files_from_dir, load_nii_image, load_dcm_image


class DataLoader(tf.keras.utils.Sequence):
    def __init__(
        self,
        directories: List[Tuple[int, str]],
        batch_size: int,
        inputs: Dict[int, np.ndarray],
        targets: Dict[int, np.ndarray],
        preprocessing: Preprocessing,
    ):
        self._batch_size = batch_size
        self._directories = directories
        self._inputs = inputs
        self._targets = targets
        self._preprocessing = preprocessing

    def __len__(self):
        return math.ceil(len(self._directories) / self._batch_size)

    def __getitem__(self, index):
        indices = self._directories[
            index * self._batch_size : (index + 1) * self._batch_size
        ]

        x = tf.convert_to_tensor(
            [
                self._preprocessing(
                    self._get_input_at_index(idx),
                    "input",
                )
                for idx, _path in indices
            ]
        )
        y = tf.convert_to_tensor(
            [
                self._preprocessing(self._get_target_at_index(idx), "target")
                for idx, _path in indices
            ]
        )
        return x, y

    def debug_directory(self, directory_index):
        num_slices = self._inputs[directory_index].shape[-1]
        for i in range(num_slices):
            yield (
                self._get_input_at_index(directory_index),
                self._get_target_at_index(directory_index),
            )

    def _get_input_at_index(self, directory_index):
        return tf.stack(
            [
                load_dcm_image(image_path).T
                for image_path in self._inputs[directory_index]
            ],
            axis=2,
        )

    def _get_target_at_index(self, directory_index):
        return load_nii_image(self._targets[directory_index])


class DataLoaderFactory:
    def __init__(
        self,
        patients_inputs_dir: str,
        patients_target_dir: str,
        input_file_ext: str,
        target_file_ext: str,
        input_shape: List[int],
        num_classes: int,
        batch_size: int,
        random_state: int,
        **kwargs,
    ):
        self.batch_size = batch_size
        self._random_state = random_state
        self._preprocessing = Preprocessing(input_shape, num_classes)
        self._directories = [
            (directory_index, directory_path.path)
            for directory_index, directory_path in enumerate(
                filter(
                    lambda x: os.path.isdir(x),
                    sorted(os.scandir(patients_inputs_dir), key=lambda x: x.path),
                )
            )
        ]
        self._inputs = {
            directory_index: [
                image_path
                for image_path in sorted(
                    get_files_from_dir(patient_input, input_file_ext)
                )
            ]
            for directory_index, patient_input in self._directories
        }

        self._targets = {
            directory_index: target_batch_path
            for directory_index, target_batch_path in enumerate(
                sorted(get_files_from_dir(patients_target_dir, target_file_ext))
            )
        }

    def produce_loaders(self, test_size: float = None, val_size: float = None):
        if not test_size:
            return self._get_loader(
                shuffle(self._directories, random_state=self._random_state)
            )

        train, test = train_test_split(
            self._directories, test_size=test_size, random_state=self._random_state
        )
        if not val_size:
            return self._get_loader(train), self._get_loader(test)

        train, val = train_test_split(
            train, test_size=val_size, random_state=self._random_state
        )
        return self._get_loader(train), self._get_loader(val), self._get_loader(test)

    def _get_loader(self, dirs):
        return DataLoader(
            dirs, self.batch_size, self._inputs, self._targets, self._preprocessing
        )
