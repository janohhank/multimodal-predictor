import numpy as np
import torch.multiprocessing as mp
import sklearn.metrics as sk_metrics
import torch
import torch.package

from pe_dataset_loader import PEDatasetLoader
from pe_net.pe_net_model_helper import PENetModelHelper


def evaluate():
    print("Starting the evaluation of late-fusion PE multimodal predictor.")

    model_package: str = "model"
    model_resource: str = "model.pkl"
    model_path: str = "d:\\Downloads\\penet_whole.pt"

    dataset_path: str = "d:\\Downloads\\radfusion-dataset\\split\\test"
    window_size: int = 24

    # Load PE-NET model
    pe_net_model = PENetModelHelper.load_model(
        model_path, model_package, model_resource
    )
    # Load PE logistic regression model
    # TBD

    # Load test dataset
    dataset_loader: PEDatasetLoader = PEDatasetLoader(
        dataset_path,
        window_size=window_size,
        num_workers=4,
    )
    print("Dataset initialized.")

    device = torch.device("cpu")
    print(f"Device initialized: {device}.")

    ground_truth_labels = {}
    pe_net_predictions = {}
    with torch.no_grad():
        for inputs, label, patient_id in dataset_loader:
            inputs = inputs.to(device)  # (1, 1, D, H, W)
            outputs = pe_net_model(inputs)
            probs = torch.sigmoid(outputs).cpu().item()

            idx = patient_id.item()
            if idx not in pe_net_predictions:
                pe_net_predictions[idx] = []
            pe_net_predictions[idx].append(probs)
            ground_truth_labels[idx] = label.item()

    final_predictions = {pid: max(probs) for pid, probs in pe_net_predictions.items()}

    max_probs, labels = np.array(list(final_predictions.values())), np.array(
        list(ground_truth_labels.values())
    )
    metrics = {
        "PE-NET AUC-PRC": sk_metrics.average_precision_score(labels, max_probs),
        "PE-NET AUC-ROC": sk_metrics.roc_auc_score(labels, max_probs),
    }
    print(metrics)

    print("Finished the evaluation of late-fusion PE multimodal predictor.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    evaluate()
