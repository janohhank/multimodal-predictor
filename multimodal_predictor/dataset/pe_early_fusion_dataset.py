import json
import os

import cv2
import numpy as np
import pandas
import torch
from pandas import DataFrame
from torch.utils.data import Dataset


class PEEarlyFusionDataset(Dataset):
    CROP_SHAPE: list = [208, 208]
    RESIZE_SHAPE: list = [224, 224]

    CONTRAST_HU_MIN: float = -100
    CONTRAST_HU_MAX: float = 900
    CONTRAST_HU_MEAN: float = 0.15897

    __window_size: int = None
    __samples: list = []

    def __init__(
        self,
        data_dir: str,
        window_size: int = 24,
    ):
        self.__window_size = window_size

        # This is a list of tuples: (npy_path, json_path, csv_path, start_idx).
        self.__samples = []

        # Match all .npy/.json/.csv files and index window positions.
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".npy") and "_ct_features" not in file_name:
                base_name = file_name[:-4]
                npy_path = os.path.join(data_dir, base_name + ".npy")
                json_path = os.path.join(data_dir, base_name + ".json")
                csv_path = os.path.join(data_dir, base_name + ".csv")

                if not os.path.exists(json_path):
                    raise FileNotFoundError(f"Missing JSON for {npy_path}")

                if not os.path.exists(json_path):
                    raise FileNotFoundError(f"Missing CSV for {npy_path}")

                volume = np.load(npy_path)
                num_slices = volume.shape[0]

                if self.__window_size > 0:
                    for start_idx in range(0, num_slices, self.__window_size):
                        self.__samples.append(
                            (npy_path, json_path, csv_path, start_idx)
                        )
                else:
                    self.__samples.append((npy_path, json_path, csv_path, 0))

    def __len__(self):
        return len(self.__samples)

    def __getitem__(self, idx: int):
        npy_path, json_path, csv_path, start_idx = self.__samples[idx]

        # Load the current sample metadata.
        with open(json_path, "r") as f:
            meta: dict = json.load(f)
        patient_id: int = int(meta["idx"])
        label: int = int(meta["label"])

        # Load the current sample EHR data.
        ehr_data: DataFrame = pandas.read_csv(csv_path)

        # Load current sample 3D CT volume (window size).
        volume = np.load(npy_path)
        if self.__window_size > 0:
            end_idx: int = min(start_idx + self.__window_size, volume.shape[0])
        else:
            end_idx: int = volume.shape[0]

        # Pad CT volume with air (-1000 HU) if needed
        if end_idx - start_idx < self.__window_size and self.__window_size > 0:
            padding = np.full(
                (self.__window_size - (end_idx - start_idx), *volume.shape[1:]),
                -1000.0,
                dtype=np.float32,
            )
            volume_window = np.concatenate([volume[start_idx:end_idx], padding], axis=0)
        else:
            volume_window = volume[start_idx:end_idx]

        # Transform CT volume.
        volume_window = self.__transform_ct_volume(volume_window)

        return (
            volume_window,
            ehr_data.drop("label", axis=1).drop("idx", axis=1).to_numpy(),
            label,
            patient_id,
        )

    def __transform_ct_volume(self, volume):
        # Rescale
        inputs = self.__resize_slice_wise(
            volume, tuple(PEEarlyFusionDataset.RESIZE_SHAPE)
        )

        # Crop
        row_margin = max(0, inputs.shape[-2] - PEEarlyFusionDataset.CROP_SHAPE[-2])
        col_margin = max(0, inputs.shape[-1] - PEEarlyFusionDataset.CROP_SHAPE[-1])
        row = row_margin // 2
        col = col_margin // 2
        inputs = inputs[
            :,
            row : row + PEEarlyFusionDataset.CROP_SHAPE[-2],
            col : col + PEEarlyFusionDataset.CROP_SHAPE[-1],
        ]

        # Normalize raw Hounsfield Units
        inputs = self.__normalize(inputs)

        # Add channel dimension -> shape: (1, D, H, W)
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
        pixels = volume.astype(np.float32)
        pixels = (pixels - PEEarlyFusionDataset.CONTRAST_HU_MIN) / (
            PEEarlyFusionDataset.CONTRAST_HU_MAX - PEEarlyFusionDataset.CONTRAST_HU_MIN
        )
        return np.clip(pixels, 0.0, 1.0) - PEEarlyFusionDataset.CONTRAST_HU_MEAN
