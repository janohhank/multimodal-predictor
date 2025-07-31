import os

import cv2
from moviepy import ImageSequenceClip, clips_array
import argparse
import json

import numpy as np
import torch.multiprocessing as mp
import torch
import torch.package
from matplotlib import pyplot as plt
from scipy.interpolate import interpolate

from cams.grad_cam import GradCAM
from dataset.pe_late_fusion_dataset import PELateFusionDataset
from pe_logistic_regression.logistic_regression_model_helper import (
    LogisticRegressionModelHelper,
)
from dataset.pe_late_fusion_dataset_loader import (
    PELateFusionDatasetLoader,
)
from pe_net.pe_net_model_helper import PENetModelHelper


def resize(cam, input_img, interpolation="linear"):
    """Resizes a volume using factorized bilinear interpolation"""
    print(input_img.shape)
    temp_cam = np.zeros((cam.shape[0], input_img.shape[0], input_img.shape[1]))
    for dim in range(temp_cam.shape[0]):
        temp_cam[dim, :, :] = cv2.resize(
            cam[dim, :, :], dsize=(temp_cam.shape[2], temp_cam.shape[1])
        )

    if temp_cam.shape[0] == 1:
        new_cam = np.tile(temp_cam, (input_img.shape[0], 1, 1))
    else:
        new_cam = np.zeros((input_img.shape[0], temp_cam.shape[1], temp_cam.shape[2]))
        for i in range(temp_cam.shape[1] * temp_cam.shape[2]):
            y = i % temp_cam.shape[2]
            x = i // temp_cam.shape[2]
            compressed = temp_cam[:, x, y]
            labels = np.arange(compressed.shape[0], step=1)
            new_labels = np.linspace(0, compressed.shape[0] - 1, new_cam.shape[0])
            f = interpolate.interp1d(labels, compressed, kind=interpolation)
            expanded = f(new_labels)
            new_cam[:, x, y] = expanded

    return new_cam


def add_heat_map(
    pixels_np, intensities_np, alpha_img=0.33, color_map="magma", normalize=True
):
    """Add a CAM heat map as an overlay on a PNG image.

    Args:
        pixels_np: Pixels to add the heat map on top of. Must be in range (0, 1).
        intensities_np: Intensity values for the heat map. Must be in range (0, 1).
        alpha_img: Weight for image when summing with heat map. Must be in range (0, 1).
        color_map: Color map scheme to use with PyPlot.
        normalize: If True, normalize the intensities to range exactly from 0 to 1.

    Returns:
        Original pixels with heat map overlaid.
    """
    assert np.max(intensities_np) <= 1 and np.min(intensities_np) >= 0
    color_map_fn = plt.get_cmap(color_map)
    if normalize:
        intensities_np = normalize_to_image(intensities_np)
    else:
        intensities_np *= 255
    heat_map = color_map_fn(intensities_np.astype(np.uint8))
    if len(heat_map.shape) == 3:
        heat_map = heat_map[:, :, :3]
    else:
        heat_map = heat_map[:, :, :, :3]

    new_img = alpha_img * pixels_np.astype(np.float32) + (
        1.0 - alpha_img
    ) * heat_map.astype(np.float32)
    new_img = np.uint8(normalize_to_image(new_img))

    return new_img


def normalize_to_image(img):
    """Normalizes img to be in the range 0-255."""
    img -= np.amin(img)
    img /= np.amax(img) + 1e-7
    img *= 255
    return img


def normalize(volume):
    """Normalize an ndarray of raw Hounsfield Units to [-1, 1].

    Clips the values to [min, max] and scales into [0, 1],
    then subtracts the mean pixel (min, max, mean are defined in constants.py).

    Args:
        pixels: NumPy ndarray to convert. Any shape.

    Returns:
        NumPy ndarray with normalized pixels in [-1, 1]. Same shape as input.
    """
    pixels = volume.astype(np.float32)
    pixels = (pixels - PELateFusionDataset.CONTRAST_HU_MIN) / (
        PELateFusionDataset.CONTRAST_HU_MAX - PELateFusionDataset.CONTRAST_HU_MIN
    )
    return np.clip(pixels, 0.0, 1.0) - PELateFusionDataset.CONTRAST_HU_MEAN


def evaluate(test_parameters_path: str):
    print("Starting the evaluation of late-fusion PE multimodal predictor.")

    with open(test_parameters_path, "r") as config_file:
        config: dict = json.load(config_file)

    dataset_path: str = config["test_dataset"]

    # Load PE-NET model
    pe_net_model_package: str = config["pe_net_model"]["model_package"]
    pe_net_model_resource: str = config["pe_net_model"]["model_resource"]
    pe_net_model_path: str = config["pe_net_model"]["model_path"]
    pe_net_window_size: int = config["pe_net_model"]["window_size"]
    pe_net_model = PENetModelHelper.load_model(
        pe_net_model_path, pe_net_model_package, pe_net_model_resource
    )

    # Load PE logistic regression model
    ehr_model_path: str = config["ehr_model"]["model_path"]
    ehr_scaler_path: str = config["ehr_model"]["scaler_path"]
    ehr_features: list = config["ehr_model"]["features"]
    logistic_regression_model = LogisticRegressionModelHelper.load_model(ehr_model_path)

    # Load test dataset
    dataset_loader: PELateFusionDatasetLoader = PELateFusionDatasetLoader(
        dataset_path,
        ehr_scaler_path,
        ehr_features,
        window_size=pe_net_window_size,
        num_workers=4,
    )
    print("Dataset initialized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device initialized: {device}.")

    grad_cam = GradCAM(pe_net_model, device, is_binary=True, is_3d=True)

    for ct_inputs, ehr_data, label, patient_id in dataset_loader:
        if label.item() != 1:
            continue

        print(f"CT inputs size: {ct_inputs.shape}")

        # ct_inputs = ct_inputs.to(device)  # (1, 1, D, H, W)

        print("Generating CAM...")
        with torch.set_grad_enabled(True):
            probs, idx = grad_cam.forward(ct_inputs)
            grad_cam.backward(idx=idx[0])  # Just take top prediction
            cam = grad_cam.get_cam("module.encoders.3")

        print("Overlaying CAM...")
        print(cam.shape)
        new_cam = resize(cam, ct_inputs.cpu().numpy()[0])
        print(new_cam.shape)

        input_np = normalize(ct_inputs.cpu().numpy()[0])
        print(input_np.shape)
        input_np = np.transpose(input_np, (1, 2, 3, 0))
        input_frames = list(input_np)

        input_normed = np.float32(input_np) / 255
        cam_frames = list(add_heat_map(input_normed, new_cam))

        output_path_input = os.path.join(
            "{}_input_fn_intermountain.gif".format(patient_id.item()),
        )
        output_path_cam = os.path.join(
            "{}_cam_fn_intermountain.gif".format(patient_id.item()),
        )
        output_path_combined = os.path.join(
            "{}_combined_fn_intermountain.gif".format(patient_id.item()),
        )

        input_clip = ImageSequenceClip(input_frames, fps=4)
        input_clip.write_gif(output_path_input, verbose=False)
        cam_clip = ImageSequenceClip(cam_frames, fps=4)
        cam_clip.write_gif(output_path_cam, verbose=False)
        combined_clip = clips_array([[input_clip, cam_clip]])
        combined_clip.write_gif(output_path_combined, verbose=False)
        break

    print("Finished the evaluation of late-fusion PE multimodal predictor.")


def main():
    parser = argparse.ArgumentParser(description="Multimodel predictor testing.")
    parser.add_argument(
        "--test_parameters_path",
        type=str,
        required=True,
        help="Test parameters descriptor (JSON) file.",
    )

    args = parser.parse_args()

    evaluate(args.test_parameters_path)


if __name__ == "__main__":
    main()
