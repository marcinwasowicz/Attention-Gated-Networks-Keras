import os
import math
from typing import List, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

from .augmentation import Augmentation
from .utils import get_files_from_dir, load_nii_image, load_dcm_image


class DataLoader(tf.keras.utils.Sequence):
    def __init__(
        self,
        directories: List[Tuple[int, str]],
        batch_size: int,
        inputs: Dict[int, np.ndarray],
        targets: Dict[int, np.ndarray],
        augmentation: Augmentation,
    ):
        self._batch_size = batch_size
        self._directories = directories
        self._inputs = inputs
        self._targets = targets
        self._augmentation = augmentation

    def __len__(self):
        return math.ceil(len(self._directories) / self._batch_size)

    def __getitem__(self, index):
        indices = self._directories[
            index * self._batch_size : (index + 1) * self._batch_size
        ]

        x = np.array([self._augmentation(self._inputs[i[0]], "input") for i in indices])
        y = np.array(
            [self._augmentation(self._targets[i[0]], "target") for i in indices]
        )
        return x, y


class DataLoaderFactory:
    def __init__(
        self,
        patients_inputs_dir: str,
        patients_target_dir: str,
        input_file_ext: str,
        target_file_ext: str,
        augmentation_args: Dict[str, any],
        batch_size: int,
        random_state: int,
    ):
        self.batch_size = batch_size
        self._random_state = random_state
        self._augmentation = Augmentation(**augmentation_args)
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
            directory_index: np.stack(
                [
                    load_dcm_image(image_path).T
                    for image_path in sorted(
                        get_files_from_dir(patient_input, input_file_ext)
                    )
                ],
                axis=2,
            )
            for directory_index, patient_input in self._directories
        }

        self._targets = {
            directory_index: load_nii_image(target_batch_path)
            for directory_index, target_batch_path in enumerate(
                sorted(get_files_from_dir(patients_target_dir, target_file_ext))
            )
        }

    def debug_directory(self, directory_index):
        num_slices = self._inputs[directory_index].shape[-1]
        for i in range(num_slices):
            yield (
                self._inputs[directory_index][:, :, i],
                self._targets[directory_index][:, :, i],
            )

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
            dirs, self.batch_size, self._inputs, self._targets, self._augmentation
        )
