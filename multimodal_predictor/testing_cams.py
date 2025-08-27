import os
from datetime import datetime

import argparse
import json

import numpy as np
import torch.nn.functional as F
import torch
import torch.package
from matplotlib import pyplot as plt
from moviepy import ImageSequenceClip, clips_array

from dataset.pe_early_fusion_dataset_loader import (
    PEEarlyFusionDatasetLoader,
)
from pe_net.pe_net_model_helper import PENetModelHelper


class GradCAM3D:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        self.gradients = None

        # Register hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        # Forward pass
        output, _ = self.model(input_tensor)

        # If no class index provided, take predicted
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # Compute weights
        gradients = self.gradients[0]  # (C, D, H, W)
        activations = self.activations[0]  # (C, D, H, W)
        weights = gradients.mean(dim=(1, 2, 3))  # (C,)

        cam = torch.zeros_like(activations[0])
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)  # ReLU
        cam = cam.detach().cpu().numpy()

        # Normalize CAM
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam


def find_layer(target_layer, model):
    for name, module in model.named_modules():
        if name == target_layer:
            return module
    raise ValueError("Invalid layer name: {}".format(target_layer))


def overlay_cam_on_slice(slice_img, cam_slice, alpha=0.4, cmap="jet"):
    slice_img = (slice_img - slice_img.min()) / (
        slice_img.max() - slice_img.min() + 1e-8
    )
    cam_slice = (cam_slice - cam_slice.min()) / (
        cam_slice.max() - cam_slice.min() + 1e-8
    )

    cam_resized = (
        F.interpolate(
            torch.tensor(cam_slice).unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            size=(208, 208),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )

    # Normalize to [0,1]
    cam_resized = (cam_resized - cam_resized.min()) / (
        cam_resized.max() - cam_resized.min() + 1e-8
    )

    cmap_func = plt.get_cmap(cmap)  # e.g. cmap = "jet"
    heatmap = cmap_func(cam_resized)[:, :, :3]  # get RGB

    # heatmap = cm.get_cmap(cmap)(cam_slice)[:, :, :3]  # RGB
    overlay = (1 - alpha) * np.repeat(
        slice_img[:, :, None], 3, axis=2
    ) + alpha * heatmap
    return np.clip(overlay, 0, 1)


def to_rgb(arr):
    if arr.ndim == 2:  # grayscale → RGB
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 1:  # single channel → RGB
        arr = np.repeat(arr, 3, axis=2)
    return arr


def evaluate(test_parameters_path: str):
    print("Starting the evaluation of late-fusion PE multimodal predictor.")
    training_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(training_datetime)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device initialized: {device}.")
    pe_net_model.to(device)
    pe_net_model.eval()

    grad_cam = GradCAM3D(
        pe_net_model,
        target_layer=find_layer("module.encoders.3", pe_net_model),
        device=device,
    )

    # Load test dataset
    dataset_loader: PEEarlyFusionDatasetLoader = PEEarlyFusionDatasetLoader(
        dataset_path,
        window_size=pe_net_window_size,
        num_workers=4,
    )
    print("Dataset initialized.")

    a: int = 0
    for ct_inputs, ehr_column_names, ehr_data, label, patient_id in dataset_loader:
        if label == 0:
            continue

        ct_inputs = ct_inputs.to(device).requires_grad_(True)  # (1, 1, D, H, W)

        # Generate CAM
        cam_volume = grad_cam.generate_cam(ct_inputs, class_idx=None)  # (D, H, W)

        volume_np = ct_inputs[0, 0].detach().cpu().numpy()  # (D, H, W)
        overlay_slices = []
        for i in range(cam_volume.shape[0]):
            overlay = overlay_cam_on_slice(volume_np[i], cam_volume[i])
            overlay_slices.append(overlay)
            # plt.imshow(overlay)
            # plt.title(f"Slice {i}")
            # plt.axis("off")
            # plt.show()

        # Suppose overlay_slices is a list of RGB numpy arrays, shape (H, W, 3), values in [0,1]
        # Convert to uint8 [0,255]
        frames_uint8 = [
            (np.clip(img * 255, 0, 255)).astype(np.uint8) for img in overlay_slices
        ]

        # Create GIF (fps controls speed)
        overlay_clip = ImageSequenceClip(frames_uint8, fps=2)

        overlay_clip.write_gif(
            os.path.join(
                training_datetime, f"{str(patient_id.item())}-{a}-gradcam_ct.gif"
            ),
            fps=2,
        )
        a = a + 1

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
