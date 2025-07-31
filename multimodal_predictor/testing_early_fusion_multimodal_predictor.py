import argparse
import json

import joblib
import numpy as np
import torch.multiprocessing as mp
import torch
import torch.package
from sklearn.metrics import average_precision_score, roc_auc_score

from plot_utils import PlotUtility
from pe_early_fusion_dataset_loader import (
    PEEarlyFusionDatasetLoader,
)
from pe_xgboost.xgboost_model_helper import XGBoostModelHelper
from pe_net.pe_net_model_helper import PENetModelHelper


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

    # Load XGBoost model
    ehr_model_path: str = config["ehr_model"]["model_path"]
    ehr_scaler_path: str = config["ehr_model"]["scaler_path"]
    ehr_selector_path: list = config["ehr_model"]["selector_path"]
    xgboost_model = XGBoostModelHelper.load_model(ehr_model_path)
    scaler = joblib.load(ehr_scaler_path)
    selector = joblib.load(ehr_selector_path)

    selected_indices = selector.get_support(indices=True)
    print(f"Selected feature indices: {selected_indices}")
    print(f"Number of selected features: {len(selected_indices)}")

    # Load test dataset
    dataset_loader: PEEarlyFusionDatasetLoader = PEEarlyFusionDatasetLoader(
        dataset_path,
        window_size=pe_net_window_size,
        num_workers=4,
    )
    print("Dataset initialized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device initialized: {device}.")

    probabilities = {}
    ground_truth_labels = {}
    with torch.no_grad():
        for ct_inputs, ehr_data, label, patient_id in dataset_loader:
            ct_inputs = ct_inputs.to(device)  # (1, 1, D, H, W)
            _, features = pe_net_model.forward(ct_inputs)

            scaled = scaler.transform(
                np.hstack([ehr_data.cpu().numpy()[0], features.cpu().numpy()])
            )
            xgboost_input = selector.transform(scaled)

            y_prob = xgboost_model.predict_proba(xgboost_input)[:, 1]

            idx = patient_id.item()
            probabilities[idx] = y_prob
            ground_truth_labels[idx] = label.item()

    predictions = {idx: int(prob > 0.5) for idx, prob in probabilities.items()}
    model_metrics = {
        "PR-AUC": average_precision_score(
            list(ground_truth_labels.values()), list(probabilities.values())
        ),
        "ROC-AUC": roc_auc_score(
            list(ground_truth_labels.values()), list(probabilities.values())
        ),
    }
    print(model_metrics)
    PlotUtility.plot_confusion_matrix(
        "early_fusion_confusion_matrix.pdf",
        list(ground_truth_labels.values()),
        list(predictions.values()),
    )

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
    mp.set_start_method("spawn")
    main()
