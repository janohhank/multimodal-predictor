import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import torch
import torch.package
from sklearn.metrics import average_precision_score, roc_auc_score

from pe_logistic_regression.logistic_regression_model_helper import (
    LogisticRegressionModelHelper,
)
from utils.plot_utils import PlotUtility
from dataset.pe_early_fusion_dataset_loader import (
    PEEarlyFusionDatasetLoader,
)
from pe_net.pe_net_model_helper import PENetModelHelper


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

    # Load Logistic Regression mixed ehr+ct model
    mixed_model_path: str = config["mixed_model"]["model_path"]
    mixed_scaler_path: str = config["mixed_model"]["scaler_path"]
    mixed_selected_features: list = config["mixed_model"]["features"]
    mixed_model = LogisticRegressionModelHelper.load_model(mixed_model_path)
    scaler = joblib.load(mixed_scaler_path)

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
        for ct_inputs, ehr_column_names, ehr_data, label, patient_id in dataset_loader:
            ct_inputs = ct_inputs.to(device)  # (1, 1, D, H, W)
            _, features = pe_net_model.forward(ct_inputs)

            ehr_numpy = ehr_data.cpu().numpy()[0]

            # EHR data
            ehr_df = pd.DataFrame(
                ehr_numpy,
                columns=[t[0] for t in ehr_column_names],
                index=pd.DataFrame(ehr_numpy).index,
            )

            # CT data
            ct = np.vstack([features.cpu().numpy() for _, row in ehr_df.iterrows()])
            ct_col_names = [f"ct_{i+1}" for i in range(ct.shape[1])]
            df_ct = pd.DataFrame(ct, columns=ct_col_names, index=ehr_df.index)

            # Mixed data
            dataset_df = pd.concat(
                [ehr_df.reset_index(drop=True), df_ct.reset_index(drop=True)], axis=1
            )

            scaled = scaler.transform(dataset_df[mixed_selected_features])
            y_prob = mixed_model.predict_proba(scaled)[:, 1]

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
        os.path.join(training_datetime, "early_fusion_confusion_matrix.pdf"),
        list(ground_truth_labels.values()),
        list(predictions.values()),
    )
    PlotUtility.plot_roc_curve(
        os.path.join(training_datetime, "early_fusion_roc_curve.pdf"),
        list(ground_truth_labels.values()),
        list(probabilities.values()),
    )
    PlotUtility.plot_pr_curve(
        os.path.join(training_datetime, "early_fusion_pr_curve.pdf"),
        list(ground_truth_labels.values()),
        list(probabilities.values()),
    )

    print("Finished the evaluation of early-fusion PE multimodal predictor.")


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
