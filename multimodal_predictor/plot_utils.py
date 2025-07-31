from abc import ABC

import seaborn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


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
