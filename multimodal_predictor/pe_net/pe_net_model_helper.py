from abc import ABC
from typing import Any

import torch


class PENetModelHelper(ABC):

    @staticmethod
    def load_model(model_path: str, model_package: str, model_resource: str) -> Any:
        print(f"Loading pre-trained PE-NET model from {model_path}")
        importer = torch.package.PackageImporter(model_path)
        model = importer.load_pickle(model_package, model_resource)
        model.eval()
        print("PE-NET model loaded.")
        return model
