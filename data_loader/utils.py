import os

import nibabel as nib
import numpy as np
from pydicom import dcmread


def get_files_from_dir(path: str, file_ext: str):
    if os.path.isfile(path) and path.endswith(file_ext):
        return [path]
    if os.path.isfile(path):
        return []
    files = []
    for entry in os.scandir(path):
        files.extend(get_files_from_dir(entry.path, file_ext))
    return files


def load_nii_image(path: str):
    return np.squeeze(nib.load(path).get_fdata().astype(np.int16))


def load_dcm_image(path: str):
    return np.squeeze(dcmread(path).pixel_array)
