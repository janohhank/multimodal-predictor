import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PEDataset(Dataset):
    def __init__(self, data_dir, window_size=24):
        self.data_dir = data_dir
        self.window_size = window_size
        self.samples = []  # list of tuples: (npy_path, json_path, start_idx)
        self.crop_shape = [208, 208]
        self.resize_shape = [224, 224]

        # Match all .npy/.json files and index window positions
        for fname in os.listdir(data_dir):
            if fname.endswith(".npy"):
                base_name = fname[:-4]
                npy_path = os.path.join(data_dir, base_name + ".npy")
                json_path = os.path.join(data_dir, base_name + ".json")

                if not os.path.exists(json_path):
                    raise FileNotFoundError(f"Missing JSON for {npy_path}")

                volume = np.load(npy_path)
                num_slices = volume.shape[0]
                for start_idx in range(0, num_slices, window_size):
                    self.samples.append((npy_path, json_path, start_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, json_path, start_idx = self.samples[idx]

        # Load label
        with open(json_path, "r") as f:
            meta = json.load(f)

        label = int(meta["label"])
        patient_id = int(meta["idx"])

        # Load 3D volume
        volume = np.load(npy_path)
        end_idx = min(start_idx + self.window_size, volume.shape[0])

        # Pad with air (-1000 HU) if needed
        """Pad 3D volume with air on both ends to desired number of slices.
        Args:
            volume_: 3D NumPy ndarray, where slices are along depth dimension (un-normalized raw HU).
            pad_value: Constant value to use for padding.
        Returns:
            Padded volume with depth args.num_slices. Extra padding voxels have pad_value.
        """
        if end_idx - start_idx < self.window_size:
            padding = np.full(
                (self.window_size - (end_idx - start_idx), *volume.shape[1:]),
                -1000.0,
                dtype=np.float32,
            )
            volume_window = np.concatenate([volume[start_idx:end_idx], padding], axis=0)
        else:
            volume_window = volume[start_idx:end_idx]

        # Transform
        volume_window = self.transform(volume_window)

        return volume_window, label, patient_id

    def transform(self, volume):
        # Rescale
        inputs = self.__resize_slice_wise(volume, tuple(self.resize_shape))

        # Crop
        row_margin = max(0, inputs.shape[-2] - self.crop_shape[-2])
        col_margin = max(0, inputs.shape[-1] - self.crop_shape[-1])
        row = row_margin // 2
        col = col_margin // 2
        inputs = inputs[
            :, row : row + self.crop_shape[-2], col : col + self.crop_shape[-1]
        ]

        # Normalize raw Hounsfield Units
        inputs = self.__normalize(inputs)

        # Add channel dimension # Add channel dimension # shape: (1, D, H, W)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = torch.from_numpy(inputs)

        return inputs

    def __resize_slice_wise(
        self, volume, slice_shape, interpolation_method=cv2.INTER_AREA
    ):
        """Resize a volume slice-by-slice.

        Args:
            volume: Volume to resize.
            slice_shape: Shape for a single slice.
            interpolation_method: Interpolation method to pass to `cv2.resize`.

        Returns:
            Volume after reshaping every slice.
        """
        slices = list(volume)
        for i in range(len(slices)):
            slices[i] = cv2.resize(
                slices[i], slice_shape, interpolation=interpolation_method
            )
        return np.array(slices)

    def __normalize(self, volume):
        """Normalize an ndarray of raw Hounsfield Units to [-1, 1].

        Clips the values to [min, max] and scales into [0, 1],
        then subtracts the mean pixel (min, max, mean are defined in constants.py).

        Args:
            pixels: NumPy ndarray to convert. Any shape.

        Returns:
            NumPy ndarray with normalized pixels in [-1, 1]. Same shape as input.
        """
        CONTRAST_HU_MIN: float = -100
        CONTRAST_HU_MAX: float = 900
        CONTRAST_HU_MEAN: float = 0.15897
        pixels = volume.astype(np.float32)
        pixels = (pixels - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN)
        return np.clip(pixels, 0.0, 1.0) - CONTRAST_HU_MEAN
