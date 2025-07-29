from typing import Any

import numpy as np
import torch.multiprocessing as mp
import sklearn.metrics as sk_metrics
import torch
import torch.package
from torch.utils.data import DataLoader
from ct_dataset import CTDataset


def load_penet_model(model_path: str) -> Any:
    print(f"Loading pre-trained PENET model from {model_path}")
    imp = torch.package.PackageImporter(model_path)
    model = imp.load_pickle("model", "model.pkl")
    model.eval()
    print("Model loaded.")
    return model


def predict():
    print("Starting multimodal prediction.")

    model_path: str = "d:\\Downloads\\penet_whole.pt"

    dataset_path: str = "d:\\Downloads\\radfusion-dataset\\split\\test"
    batch_size: int = 1

    model = load_penet_model(model_path)

    print("Loading dataset.")
    dataset_loader: DataLoader[CTDataset] = DataLoader(
        CTDataset(dataset_path, window_size=24),
        batch_size=batch_size,
        num_workers=1,
    )
    print("Dataset initialized.")

    labels = {}
    patient_probs = {}
    device = torch.device("cpu")
    with torch.no_grad():
        for inputs, label, patient_id in dataset_loader:
            inputs = inputs.to(device)  # (1, 1, D, H, W)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().item()

            idx = patient_id.item()
            if idx not in patient_probs:
                patient_probs[idx] = []
            patient_probs[idx].append(probs)
            labels[idx] = label.item()

    final_predictions = {pid: max(probs) for pid, probs in patient_probs.items()}
    print(f"Final predictions: {final_predictions}")

    max_probs, labels = np.array(list(final_predictions.values())), np.array(
        list(labels.values())
    )
    print(max_probs)
    print(labels)
    metrics = {
        "AUPRC": sk_metrics.average_precision_score(labels, max_probs),
        "AUROC": sk_metrics.roc_auc_score(labels, max_probs),
    }
    print(metrics)

    print("Finished multimodal prediction.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    predict()
