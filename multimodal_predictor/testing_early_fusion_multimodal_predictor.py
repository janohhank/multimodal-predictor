import argparse
import json

import numpy as np
import torch.multiprocessing as mp
import torch
import torch.package

from pe_logistic_regression.logistic_regression_model_helper import (
    LogisticRegressionModelHelper,
)
from pe_dataset_loader import PEDatasetLoader
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

    # Load PE logistic regression model
    ehr_model_path: str = config["ehr_model"]["model_path"]
    ehr_scaler_path: str = config["ehr_model"]["scaler_path"]
    ehr_features: list = config["ehr_model"]["features"]
    logistic_regression_model = LogisticRegressionModelHelper.load_model(ehr_model_path)

    # Load test dataset
    dataset_loader: PEDatasetLoader = PEDatasetLoader(
        dataset_path,
        ehr_scaler_path,
        ehr_features,
        window_size=pe_net_window_size,
        num_workers=4,
    )
    print("Dataset initialized.")

    device = torch.device("cpu")
    print(f"Device initialized: {device}.")

    ground_truth_labels = {}
    pe_net_predictions = {}
    with torch.no_grad():
        for ct_inputs, ehr_data, label, patient_id in dataset_loader:
            ct_inputs = ct_inputs.to(device)  # (1, 1, D, H, W)
            logits, features = pe_net_model.forward(ct_inputs)
            probs = torch.sigmoid(logits).cpu().item()

            idx = patient_id.item()
            if idx not in pe_net_predictions:
                # 2048
                np.save(
                    str(patient_id.item()) + "_ct_features.npy",
                    features.cpu().numpy()[0],
                )
                pe_net_predictions[idx] = []
            pe_net_predictions[idx].append(probs)
            ground_truth_labels[idx] = label.item()

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
