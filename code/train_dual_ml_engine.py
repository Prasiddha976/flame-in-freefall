"""
Step 2 (Upgraded): Physics-Informed Feature Engineering & Dual ML Engine
Author: Space Apps 2026 Developer
Project: Flame in Freefall - Predictive Space Habitat Fire Safety System
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


NUMERIC_FEATURES = [
    "pressure_kpa",
    "initial_o2_pct",
    "inert_n2_pct",
    "co2_initial_pct",
    "inert_he_pct",
    "initial_size_mm",
    "co_final_ppm",
    "burning_rate_mm2_s",
    "o2_partial_pressure_kpa",
    "gas_thermal_conductivity",
    "o2_to_inert_ratio",
    "initial_size_sq_mm2",
    "theoretical_d2_burn_index",
    "toxic_smoldering_index",
    "kinematic_burn_time_est",
]

CATEGORICAL_FEATURES = ["fuel_category", "fuel_family"]


def classify_fuel_family(material_str: str) -> str:
    """Normalizes raw NASA material labels into clean chemical fuel families."""
    s = str(material_str).lower()
    if "methanol" in s:
        return "Methanol"
    if "heptane" in s:
        return "Heptane"
    if "pmma" in s or "film" in s or "thick" in s:
        return "PMMA_Acrylic"
    if "sibal" in s:
        return "SIBAL_Cotton_Fiberglass"
    if "nomex" in s:
        return "Nomex_Fabric"
    if "rod" in s:
        return "Acrylic_Rod"
    return "Wire_Polymer"


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes astro-combustion physical invariants from raw NASA PSI telemetry.

    Shared identically between training and real-time Streamlit inference.
    """
    d = df.copy()

    # 1. Dalton's Law: Oxygen Partial Pressure (kPa)
    d["o2_partial_pressure_kpa"] = d["pressure_kpa"] * (
        d["initial_o2_pct"] / 100.0
    )

    # 2. Effective Gas Mixture Thermal Conductivity (W / m*K)
    d["gas_thermal_conductivity"] = (
        (d["inert_he_pct"] / 100.0) * 0.151
        + (d["inert_n2_pct"] / 100.0) * 0.026
        + (d["co2_initial_pct"] / 100.0) * 0.0166
        + (d["initial_o2_pct"] / 100.0) * 0.0265
    )

    # 3. Oxygen-to-Inert Diluent Ratio
    inert_sum = (
        d["inert_n2_pct"] + d["inert_he_pct"] + d["co2_initial_pct"]
    ).replace(0, 1.0)
    d["o2_to_inert_ratio"] = d["initial_o2_pct"] / inert_sum

    # 4. Microgravity Droplet D^2-Law Scaling Features
    d["initial_size_sq_mm2"] = d["initial_size_mm"] ** 2
    d["theoretical_d2_burn_index"] = d["initial_size_sq_mm2"] / d[
        "o2_partial_pressure_kpa"
    ].replace(0, 1.0)

    # 5. Toxic Smoldering Index (CO ppm per % ambient O2)
    d["toxic_smoldering_index"] = d["co_final_ppm"] / d[
        "initial_o2_pct"
    ].replace(0, 1.0)

    # 6. Kinematic Residence Time Estimate (D0^2 / K for droplets, L / K for solids)
    safe_rate = d["burning_rate_mm2_s"].replace(0, 0.1)
    d["kinematic_burn_time_est"] = np.where(
        d["fuel_category"] == "Liquid Droplet",
        d["initial_size_sq_mm2"] / safe_rate,
        d["initial_size_mm"] / safe_rate,
    )

    # 7. Clean Chemical Fuel Family
    if "fuel_family" not in d.columns:
        source_col = "fuel_material" if "fuel_material" in d.columns else "fuel_category"
        d["fuel_family"] = d[source_col].apply(classify_fuel_family)

    return d


def build_preprocessor() -> ColumnTransformer:
    """Creates a leak-free Scikit-Learn preprocessing transformer."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def train_and_export_models(
    data_path: str = "organized_data/master_microgravity_fires.csv",
    model_dir: str = "models",
) -> dict:
    """Trains the upgraded Dual ML Engine on real NASA PSI telemetry and saves .pkl weights."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Cannot find '{data_path}'. Run Step 1 (organize_nasa_psi_data.py) first."
        )

    os.makedirs(model_dir, exist_ok=True)
    raw_df = pd.read_csv(data_path)
    feat_df = engineer_physics_features(raw_df)
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # =========================================================================
    # ENGINE 1: MULTI-CLASS FIRE RISK CLASSIFIER (Safe vs. Warning vs. Critical)
    # =========================================================================
    X_cls = feat_df[all_features]
    y_cls = feat_df["fire_risk_level"].astype(int)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.20, random_state=42, stratify=y_cls
    )

    if HAS_XGBOOST:
        clf_estimator = XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="mlogloss",
        )
        engine_name = "XGBoost"
    else:
        clf_estimator = RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42
        )
        engine_name = "RandomForest"

    clf_pipeline = Pipeline(
        [("preprocessor", build_preprocessor()), ("classifier", clf_estimator)]
    )

    # 5-Fold Stratified Cross-Validation across all 393 real NASA runs
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf_pipeline, X_cls, y_cls, cv=cv, scoring="accuracy")

    clf_pipeline.fit(X_train_c, y_train_c)
    y_pred_c = clf_pipeline.predict(X_test_c)
    cls_acc = float(accuracy_score(y_test_c, y_pred_c))
    cls_f1 = float(f1_score(y_test_c, y_pred_c, average="weighted"))

    print("=================================================================")
    print(f"[ENGINE 1] Fire Risk Classifier Trained ({engine_name})")
    print(f"Holdout Test Accuracy (79 runs) : {cls_acc * 100:.2f}%")
    print(f"5-Fold CV Mean Accuracy         : {cv_scores.mean() * 100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    print(f"Weighted F1 Score               : {cls_f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test_c, y_pred_c, target_names=["Safe (0)", "Warning (1)", "Critical (2)"]
        )
    )

    # =========================================================================
    # ENGINE 2: FLAME EXTINCTION THRESHOLD TIME REGRESSOR (seconds)
    # =========================================================================
    active_burn_df = feat_df[feat_df["extinction_time_s"] <= 120.0].copy()
    X_reg = active_burn_df[all_features]
    y_reg = active_burn_df["extinction_time_s"].astype(float)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )

    # RandomForestRegressor excels at sharp physical piecewise regimes (Liquid D^2 vs Solid L)
    reg_estimator = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42
    )

    reg_pipeline = Pipeline(
        [("preprocessor", build_preprocessor()), ("regressor", reg_estimator)]
    )
    reg_pipeline.fit(X_train_r, y_train_r)

    y_pred_r = np.clip(reg_pipeline.predict(X_test_r), 0.1, None)
    reg_mae = float(mean_absolute_error(y_test_r, y_pred_r))
    reg_r2 = float(r2_score(y_test_r, y_pred_r))

    print("=================================================================")
    print("[ENGINE 2] Extinction Time Regressor Trained (Physics-Guided RF)")
    print(f"Mean Absolute Error (MAE): {reg_mae:.3f} seconds")
    print(f"R-Squared (R2) Score     : {reg_r2:.4f}")
    print("=================================================================")

    # Export serialized weights & metrics
    clf_path = os.path.join(model_dir, "fire_risk_classifier.pkl")
    reg_path = os.path.join(model_dir, "extinction_time_regressor.pkl")
    metrics_path = os.path.join(model_dir, "model_metrics.json")

    joblib.dump(clf_pipeline, clf_path)
    joblib.dump(reg_pipeline, reg_path)

    metrics_summary = {
        "engine_backend": engine_name,
        "total_nasa_runs": int(len(feat_df)),
        "classifier_accuracy": round(cls_acc, 4),
        "classifier_cv_accuracy": round(float(cv_scores.mean()), 4),
        "classifier_weighted_f1": round(cls_f1, 4),
        "regressor_mae_seconds": round(reg_mae, 3),
        "regressor_r2": round(reg_r2, 4),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"[+] Saved Classifier Weights : {clf_path}")
    print(f"[+] Saved Regressor Weights  : {reg_path}")
    print(f"[+] Saved Evaluation Metrics : {metrics_path}")
    return metrics_summary


if __name__ == "__main__":
    train_and_export_models()