from abc import ABC

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


class PlotUtility(ABC):

    @staticmethod
    def plot_confusion_matrix(
        file_path: str,
        y_test,
        y_pred,
    ):
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Negative", "Positive"]  # Adjust as needed

        plt.figure(figsize=(5, 4))
        seaborn.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        plt.savefig(
            file_path,
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()

    @staticmethod
    def plot_roc_curve(
        file_path: str,
        y_test,
        y_probs,
    ):
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        balanced_idx = np.argmin(np.abs(tpr + fpr - 1))
        balanced_fpr = fpr[balanced_idx]
        balanced_tpr = tpr[balanced_idx]

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        plt.scatter(
            balanced_fpr,
            balanced_tpr,
            color="green",
            marker="s",
            label=f"Balanced FPR/TPR = {balanced_tpr:.4f}",
            zorder=5,
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            file_path,
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()

    @staticmethod
    def plot_pr_curve(
        file_path: str,
        y_test,
        y_probs,
    ):
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        ap_score = average_precision_score(y_test, y_probs)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]

        balance_idx = np.argmin(np.abs(precision - recall))
        balanced_precision = precision[balance_idx]
        balanced_recall = recall[balance_idx]

        plt.figure(figsize=(6, 5))
        plt.plot(
            recall, precision, label=f"PR Curve (AP = {ap_score:.4f})", color="blue"
        )
        plt.scatter(
            optimal_recall,
            optimal_precision,
            color="red",
            label=f"Max F1 = {optimal_f1:.4f}",
            zorder=5,
        )
        plt.scatter(
            balanced_recall,
            balanced_precision,
            color="green",
            marker="s",
            label=f"Balanced P/R = {balanced_precision:.4f}",
            zorder=5,
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            file_path,
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()
