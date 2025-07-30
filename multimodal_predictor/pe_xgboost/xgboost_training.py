import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# === CONFIGURATION ===
CT_FEATURE_DIR = "D:\\Downloads\\radfusion-dataset\\split\\train"
CT_FEATURE_DIM = 2048

# === LOAD EHR DATA ===
ehr_df = pd.read_csv(
    "D:\\RoutePatch\\GitWorking\\multi-objective-regression\\test_data\\rad_fusion_train_modified.csv"
)


# === LOAD CT FEATURES ===
def load_ct_features(idx):
    filepath = (
        f"D:\\Downloads\\radfusion-dataset\\split\\train\\{int(idx)}_ct_features.npy"
    )
    return np.load(filepath)


# Create X and y
X_ct = np.vstack([load_ct_features(row["idx"]) for _, row in ehr_df.iterrows()])
X_ehr = ehr_df.drop("label", axis=1).drop("idx", axis=1).to_numpy()
X = np.hstack([X_ehr, X_ct])
y = ehr_df["label"].values

# === PIPELINE ===
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "feature_select",
            SelectKBest(score_func=f_classif, k=100),
        ),  # choose K appropriately
        ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
    ]
)

# === HYPERPARAMETER SPACE ===
param_distributions = {
    "feature_select__k": [50, 100, 200, 300],
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [3, 5, 7, 10],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__subsample": [0.6, 0.8, 1.0],
    "classifier__colsample_bytree": [0.6, 0.8, 1.0],
}

# === HYPERPARAMETER TUNING ===
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=25,
    scoring="f1",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

search.fit(X, y)

# === EVALUATION ===
print("Best params:", search.best_params_)
# y_pred = search.predict(X_test)
# print(classification_report(y_test, y_pred))


# Save components separately if needed
scaler = search.best_estimator_.named_steps["scaler"]
selector = search.best_estimator_.named_steps["feature_select"]
xgb_model = search.best_estimator_.named_steps["classifier"]

# Optionally save each one individually
joblib.dump(scaler, "scaler.joblib")
joblib.dump(selector, "selector.joblib")
joblib.dump(xgb_model, "xgb_model.joblib")
