from abc import ABC
from typing import Any

import joblib


class LogisticRegressionModelHelper(ABC):

    @staticmethod
    def load_model(model_path: str) -> Any:
        return joblib.load(model_path)
