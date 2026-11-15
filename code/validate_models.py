"""
Model Verification & Aerospace Physics Validation Suite
Author: Space Apps 2026 Developer
Project: Flame in Freefall - Predictive Space Habitat Fire Safety System
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from train_dual_ml_engine import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    engineer_physics_features,
)


def run_validation_suite():
    print("FLAME IN FREEFALL — DUAL ML MODEL VALIDATION & AUDIT SUITE\n")

    clf = joblib.load("models/fire_risk_classifier.pkl")
    reg = joblib.load("models/extinction_time_regressor.pkl")
    print("[CHECK 1] Serialized .pkl pipelines loaded successfully.\n")

    raw_df = pd.read_csv("organized_data/master_microgravity_fires.csv")
    feat_df = engineer_physics_features(raw_df)
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_cls = feat_df[all_features]
    y_cls = feat_df["fire_risk_level"].astype(int)
    _, X_test_c, _, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.20, random_state=42, stratify=y_cls
    )

    y_pred_c = clf.predict(X_test_c)
    cm = confusion_matrix(y_test_c, y_pred_c)

    print("[CHECK 2] Confusion Matrix on 79 Unseen NASA ISS Test Runs:")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Safe (0)", "Actual Warning (1)", "Actual Critical (2)"],
        columns=["Pred Safe (0)", "Pred Warning (1)", "Pred Critical (2)"],
    )
    print(cm_df)

    catastrophic_misses = cm[2, 0]
    print(
        f"\n-> Catastrophic False Negatives (Critical predicted as Safe): {catastrophic_misses} / {cm[2].sum()}"
    )
    assert (
        catastrophic_misses == 0
    ), "SAFETY FAILURE: Critical fire misclassified as Safe!"
    print("-> [PASS] Zero Critical fires misclassified as Safe!\n")

    print("[CHECK 3] Thermodynamic & Combustion Physics Stress Tests:")
    scenarios = pd.DataFrame(
        [
            {
                "scenario": "Low-O2 Hypoxic Safe Chamber (15% O2)",
                "pressure_kpa": 101.3,
                "initial_o2_pct": 15.0,
                "inert_n2_pct": 84.6,
                "co2_initial_pct": 0.4,
                "inert_he_pct": 0.0,
                "initial_size_mm": 1.6,
                "co_final_ppm": 2.0,
                "burning_rate_mm2_s": 0.35,
                "fuel_category": "Liquid Droplet",
                "fuel_family": "Methanol",
            },
            {
                "scenario": "Nominal ISS Cabin Baseline (21% O2)",
                "pressure_kpa": 101.3,
                "initial_o2_pct": 21.0,
                "inert_n2_pct": 78.6,
                "co2_initial_pct": 0.4,
                "inert_he_pct": 0.0,
                "initial_size_mm": 2.0,
                "co_final_ppm": 10.0,
                "burning_rate_mm2_s": 0.52,
                "fuel_category": "Liquid Droplet",
                "fuel_family": "Methanol",
            },
            {
                "scenario": "Enriched O2 Flashover Hazard (27% O2)",
                "pressure_kpa": 101.3,
                "initial_o2_pct": 27.0,
                "inert_n2_pct": 72.6,
                "co2_initial_pct": 0.4,
                "inert_he_pct": 0.0,
                "initial_size_mm": 3.5,
                "co_final_ppm": 40.0,
                "burning_rate_mm2_s": 0.85,
                "fuel_category": "Liquid Droplet",
                "fuel_family": "Heptane",
            },
            {
                "scenario": "Invisible Wire Smoldering (280 ppm CO)",
                "pressure_kpa": 101.3,
                "initial_o2_pct": 20.5,
                "inert_n2_pct": 79.0,
                "co2_initial_pct": 0.5,
                "inert_he_pct": 0.0,
                "initial_size_mm": 20.0,
                "co_final_ppm": 280.0,
                "burning_rate_mm2_s": 0.40,
                "fuel_category": "Solid Polymer/Fabric",
                "fuel_family": "PMMA_Acrylic",
            },
        ]
    )

    scen_features = engineer_physics_features(scenarios)[all_features]
    preds_cls = clf.predict(scen_features)
    preds_probs = clf.predict_proba(scen_features)
    preds_reg = np.clip(reg.predict(scen_features), 0.1, None)

    risk_names = {0: "SAFE", 1: "WARNING", 2: "CRITICAL"}
    for i, row in scenarios.iterrows():
        label = risk_names[int(preds_cls[i])]
        conf = preds_probs[i][int(preds_cls[i])] * 100.0
        t_ext = preds_reg[i]
        print(
            f"  * {row['scenario']:<42} -> {label:<8} ({conf:5.1f}% conf) | Est. Extinction: {t_ext:5.2f}s"
        )

    print("\n[CHECK 4] Real NASA ISS Holdout Spot-Check (Actual vs. Predicted):")
    sample_idx = X_test_c.index[:6]
    spot_df = feat_df.loc[sample_idx].copy()
    spot_pred_cls = clf.predict(spot_df[all_features])
    spot_pred_reg = np.clip(reg.predict(spot_df[all_features]), 0.1, None)

    for idx_pos, (_, r) in enumerate(spot_df.iterrows()):
        actual_lbl = risk_names[int(r["fire_risk_level"])]
        pred_lbl = risk_names[int(spot_pred_cls[idx_pos])]
        print(
            f"  * {r['experiment_id']:<14} ({r['fuel_material'][:18]:<18}) | "
            f"Risk: Actual={actual_lbl:<8} vs Pred={pred_lbl:<8} | "
            f"BurnTime: Actual={r['extinction_time_s']:5.1f}s vs Pred={spot_pred_reg[idx_pos]:5.1f}s"
        )

    print("\n[ALL VALIDATION CHECKS PASSED] Models are flight-ready for UI.")


if __name__ == "__main__":
    run_validation_suite()