import argparse
import json

import numpy as np
import torch.multiprocessing as mp
import sklearn.metrics as sk_metrics
import torch
import torch.package
from sklearn.metrics import roc_auc_score, average_precision_score

from plot_utils import PlotUtility
from pe_logistic_regression.logistic_regression_model_helper import (
    LogisticRegressionModelHelper,
)
from pe_late_fusion_dataset_loader import PELateFusionDatasetLoader
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

    ground_truth_labels = {}
    pe_net_probabilities = {}
    ehr_model_probabilities = {}
    with torch.no_grad():
        for ct_inputs, ehr_data, label, patient_id in dataset_loader:
            ct_inputs = ct_inputs.to(device)  # (1, 1, D, H, W)
            logits, _ = pe_net_model(ct_inputs)
            probs = torch.sigmoid(logits).cpu().item()

            idx = patient_id.item()
            if idx not in pe_net_probabilities:
                pe_net_probabilities[idx] = []

                y_probs = logistic_regression_model.predict_proba(
                    ehr_data.cpu().numpy()[0]
                )[:, 1]
                ehr_model_probabilities[idx] = y_probs
            pe_net_probabilities[idx].append(probs)
            ground_truth_labels[idx] = label.item()

    final_pe_net_probabilities = {
        pid: max(probs) for pid, probs in pe_net_probabilities.items()
    }
    final_pe_net_predictions = {
        idx: int(prob > 0.5) for idx, prob in final_pe_net_probabilities.items()
    }

    max_probs, labels = np.array(list(final_pe_net_probabilities.values())), np.array(
        list(ground_truth_labels.values())
    )
    pe_net_metrics = {
        "PE-Net: PR-AUC": sk_metrics.average_precision_score(labels, max_probs),
        "PE-Net: ROC-AUC": sk_metrics.roc_auc_score(labels, max_probs),
    }
    print(pe_net_metrics)
    PlotUtility.plot_confusion_matrix(
        "pe_net_only_confusion_matrix.pdf",
        labels,
        list(final_pe_net_predictions.values()),
    )

    ehr_model_predictions = {
        idx: int(prob > 0.5) for idx, prob in ehr_model_probabilities.items()
    }
    ehr_model_metrics = {
        "EHR Model: PR-AUC": average_precision_score(
            labels, list(ehr_model_probabilities.values())
        ),
        "EHR MOdel: ROC-AUC": roc_auc_score(
            labels, list(ehr_model_probabilities.values())
        ),
    }
    print(ehr_model_metrics)
    PlotUtility.plot_confusion_matrix(
        "ehr_only_confusion_matrix.pdf", labels, list(ehr_model_predictions.values())
    )

    final_probabilities_max = {}
    for idx, pe_net_prediction in final_pe_net_probabilities.items():
        ehr_model_prediction = ehr_model_probabilities[idx]
        final_probabilities_max[idx] = max(pe_net_prediction, ehr_model_prediction[0])
    final_predictions_max = {
        idx: int(prob > 0.5) for idx, prob in final_probabilities_max.items()
    }
    final_metrics_max = {
        "FINAL-MAX PR-AUC": average_precision_score(
            labels, list(final_probabilities_max.values())
        ),
        "FINAL-MAX ROC-AUC": roc_auc_score(
            labels, list(final_probabilities_max.values())
        ),
    }
    print(final_metrics_max)
    PlotUtility.plot_confusion_matrix(
        "late_fusion_max_confusion_matrix.pdf",
        labels,
        list(final_predictions_max.values()),
    )

    final_probabilities_avg = {}
    for idx, pe_net_prediction in final_pe_net_probabilities.items():
        ehr_model_prediction = ehr_model_probabilities[idx]
        final_probabilities_avg[idx] = (
            pe_net_prediction + ehr_model_prediction[0]
        ) / 2.0
    final_predictions_avg = {
        idx: int(prob > 0.5) for idx, prob in final_probabilities_avg.items()
    }
    final_metrics_average = {
        "FINAL-AVG PR-AUC": average_precision_score(
            labels, list(final_probabilities_avg.values())
        ),
        "FINAL-AVG ROC-AUC": roc_auc_score(
            labels, list(final_probabilities_avg.values())
        ),
    }
    print(final_metrics_average)
    PlotUtility.plot_confusion_matrix(
        "late_fusion_avg_confusion_matrix.pdf",
        labels,
        list(final_predictions_avg.values()),
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
