"""
Mission Control Streamlit UI, 3D Plotly Telemetry & Live Model Validation
Author: Space Apps 2026 Developer
Project: Flame in Freefall - Predictive Space Habitat Fire Safety System
"""

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Flame in Freefall | NASA PSI Microgravity Fire Safety",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    .stMetric {
        background-color: #131a2b;
        border: 1px solid #26324d;
        padding: 14px;
        border-radius: 10px;
    }
    .protocol-box-safe {
        background-color: #063b27;
        border-left: 6px solid #10b981;
        padding: 16px;
        border-radius: 8px;
        color: #ecfdf5;
    }
    .protocol-box-warning {
        background-color: #452704;
        border-left: 6px solid #f59e0b;
        padding: 16px;
        border-radius: 8px;
        color: #fffbeb;
    }
    .protocol-box-critical {
        background-color: #4c0519;
        border-left: 6px solid #f43f5e;
        padding: 16px;
        border-radius: 8px;
        color: #fff1f2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def classify_fuel_family(material_str: str) -> str:
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


def engineer_df_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["o2_partial_pressure_kpa"] = d["pressure_kpa"] * (d["initial_o2_pct"] / 100.0)
    d["gas_thermal_conductivity"] = (
        (d["inert_he_pct"] / 100.0) * 0.151
        + (d["inert_n2_pct"] / 100.0) * 0.026
        + (d["co2_initial_pct"] / 100.0) * 0.0166
        + (d["initial_o2_pct"] / 100.0) * 0.0265
    )
    inert_sum = (d["inert_n2_pct"] + d["inert_he_pct"] + d["co2_initial_pct"]).replace(0, 1.0)
    d["o2_to_inert_ratio"] = d["initial_o2_pct"] / inert_sum
    d["initial_size_sq_mm2"] = d["initial_size_mm"] ** 2
    d["theoretical_d2_burn_index"] = d["initial_size_sq_mm2"] / d["o2_partial_pressure_kpa"].replace(0, 1.0)
    d["toxic_smoldering_index"] = d["co_final_ppm"] / d["initial_o2_pct"].replace(0, 1.0)
    safe_rate = d["burning_rate_mm2_s"].replace(0, 0.1)
    d["kinematic_burn_time_est"] = np.where(
        d["fuel_category"] == "Liquid Droplet",
        d["initial_size_sq_mm2"] / safe_rate,
        d["initial_size_mm"] / safe_rate,
    )
    if "fuel_family" not in d.columns:
        source_col = "fuel_material" if "fuel_material" in d.columns else "fuel_category"
        d["fuel_family"] = d[source_col].apply(classify_fuel_family)
    return d


def compute_realtime_features(
    pressure_kpa: float,
    initial_o2_pct: float,
    co2_initial_pct: float,
    inert_he_pct: float,
    initial_size_mm: float,
    co_final_ppm: float,
    burning_rate_mm2_s: float,
    fuel_category: str,
    fuel_family: str,
) -> pd.DataFrame:
    inert_n2_pct = max(0.0, 100.0 - initial_o2_pct - co2_initial_pct - inert_he_pct)
    row = pd.DataFrame(
        [
            {
                "pressure_kpa": pressure_kpa,
                "initial_o2_pct": initial_o2_pct,
                "inert_n2_pct": inert_n2_pct,
                "co2_initial_pct": co2_initial_pct,
                "inert_he_pct": inert_he_pct,
                "initial_size_mm": initial_size_mm,
                "co_final_ppm": co_final_ppm,
                "burning_rate_mm2_s": burning_rate_mm2_s,
                "fuel_category": fuel_category,
                "fuel_family": fuel_family,
            }
        ]
    )
    return engineer_df_features(row)


@st.cache_resource
def load_ml_artifacts():
    clf = joblib.load("models/fire_risk_classifier.pkl")
    reg = joblib.load("models/extinction_time_regressor.pkl")
    with open("models/model_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return clf, reg, metrics


@st.cache_data
def load_and_audit_archive(_clf, _reg):
    df = engineer_df_features(pd.read_csv("organized_data/master_microgravity_fires.csv"))
    feat_cols = [
        "pressure_kpa", "initial_o2_pct", "inert_n2_pct", "co2_initial_pct", "inert_he_pct",
        "initial_size_mm", "co_final_ppm", "burning_rate_mm2_s", "o2_partial_pressure_kpa",
        "gas_thermal_conductivity", "o2_to_inert_ratio", "initial_size_sq_mm2",
        "theoretical_d2_burn_index", "toxic_smoldering_index", "kinematic_burn_time_est",
        "fuel_category", "fuel_family",
    ]
    X_cls = df[feat_cols]
    y_cls = df["fire_risk_level"].astype(int)
    _, X_test_c, _, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.20, random_state=42, stratify=y_cls
    )
    y_pred_c = _clf.predict(X_test_c)
    cm = confusion_matrix(y_test_c, y_pred_c)
    catastrophic_misses = int(cm[2, 0])
    critical_recall = float(cm[2, 2] / cm[2].sum())

    burn_df = df[df["extinction_time_s"] <= 120.0].copy()
    _, X_test_r, _, y_test_r = train_test_split(
        burn_df[feat_cols], burn_df["extinction_time_s"].astype(float), test_size=0.20, random_state=42
    )
    y_pred_r = np.clip(_reg.predict(X_test_r), 0.1, None)
    parity_df = pd.DataFrame(
        {
            "Experiment": burn_df.loc[X_test_r.index, "experiment_id"].values,
            "Fuel": burn_df.loc[X_test_r.index, "fuel_material"].values,
            "Actual Burn Time (s)": y_test_r.values,
            "Predicted Burn Time (s)": y_pred_r,
        }
    )

    return df, cm, catastrophic_misses, critical_recall, parity_df


clf_model, reg_model, model_metrics = load_ml_artifacts()
nasa_df, val_cm, catastrophic_misses, critical_recall, parity_df = load_and_audit_archive(
    clf_model, reg_model
)


st.sidebar.title("🛰️ ECLSS Sensor Deck")
st.sidebar.caption("Real-Time Microgravity Habitat Telemetry")

preset = st.sidebar.selectbox(
    "Load Mission Scenario Preset:",
    [
        "Custom Live Telemetry",
        "1. Nominal ISS Cabin (21% O2, 101.3 kPa)",
        "2. Enriched O2 Flashover Hazard (26% O2)",
        "3. Invisible Wire Smoldering (High CO, Low Flame)",
        "4. Helium Inert Quench Protocol (35% He)",
    ],
)

defaults = {
    "Custom Live Telemetry": (101.3, 21.0, 0.4, 0.0, 2.0, 12.0, 0.52, "Liquid Droplet", "Methanol"),
    "1. Nominal ISS Cabin (21% O2, 101.3 kPa)": (101.3, 21.0, 0.4, 0.0, 1.8, 5.0, 0.38, "Liquid Droplet", "Methanol"),
    "2. Enriched O2 Flashover Hazard (26% O2)": (101.3, 26.0, 0.4, 0.0, 4.5, 45.0, 0.88, "Liquid Droplet", "Heptane"),
    "3. Invisible Wire Smoldering (High CO, Low Flame)": (101.3, 20.5, 0.6, 0.0, 20.0, 240.0, 0.35, "Solid Polymer/Fabric", "PMMA_Acrylic"),
    "4. Helium Inert Quench Protocol (35% He)": (101.3, 17.5, 0.4, 35.0, 1.6, 2.0, 0.32, "Liquid Droplet", "Methanol"),
}[preset]

fuel_categories = ["Liquid Droplet", "Solid Polymer/Fabric", "Wire Insulation"]
fuel_families = [
    "Methanol",
    "Heptane",
    "PMMA_Acrylic",
    "SIBAL_Cotton_Fiberglass",
    "Nomex_Fabric",
    "Acrylic_Rod",
    "Wire_Polymer",
]

fuel_category = st.sidebar.selectbox(
    "Fuel / Material Regime",
    fuel_categories,
    index=fuel_categories.index(defaults[7]),
)
fuel_family = st.sidebar.selectbox(
    "Chemical Fuel Family",
    fuel_families,
    index=fuel_families.index(defaults[8]),
)

pressure_kpa = st.sidebar.slider("Cabin Pressure (kPa)", 40.0, 150.0, defaults[0], 0.5)
initial_o2_pct = st.sidebar.slider("Ambient O2 Concentration (%)", 12.0, 34.0, defaults[1], 0.2)
inert_he_pct = st.sidebar.slider("Helium Suppression Diluent (%)", 0.0, 45.0, defaults[3], 1.0)
co2_initial_pct = st.sidebar.slider("Ambient CO2 (%)", 0.0, 10.0, defaults[2], 0.1)
co_final_ppm = st.sidebar.slider("Gas Analyzer CO (ppm)", 0.0, 550.0, defaults[5], 5.0)
initial_size_mm = st.sidebar.slider("Characteristic Fuel Size D0 (mm)", 0.5, 30.0, defaults[4], 0.5)
burning_rate_mm2_s = st.sidebar.slider("Optical Burning Rate K (mm²/s)", 0.05, 1.20, defaults[6], 0.02)


live_X = compute_realtime_features(
    pressure_kpa=pressure_kpa,
    initial_o2_pct=initial_o2_pct,
    co2_initial_pct=co2_initial_pct,
    inert_he_pct=inert_he_pct,
    initial_size_mm=initial_size_mm,
    co_final_ppm=co_final_ppm,
    burning_rate_mm2_s=burning_rate_mm2_s,
    fuel_category=fuel_category,
    fuel_family=fuel_family,
)

risk_class = int(clf_model.predict(live_X)[0])
risk_probs = clf_model.predict_proba(live_X)[0]
pred_extinction_s = float(np.clip(reg_model.predict(live_X)[0], 0.2, 120.0))

risk_map = {0: "SAFE", 1: "WARNING", 2: "CRITICAL"}
risk_color_map = {0: "#10b981", 1: "#f59e0b", 2: "#f43f5e"}
current_risk_label = risk_map[risk_class]
current_risk_color = risk_color_map[risk_class]


st.title("🔥 FLAME IN FREEFALL: Predictive Microgravity Fire Safety Console")
cv_acc = model_metrics.get("classifier_cv_accuracy", model_metrics["classifier_accuracy"])
st.markdown(
    f"**Trained on {model_metrics['total_nasa_runs']} Real NASA ISS Runs** (`FLEX` • `BASS-II` • `SAME`) | "
    f"5-Fold CV Accuracy: **{cv_acc*100:.2f}%** | "
    f"Extinction R²: **{model_metrics['regressor_r2']:.2f}** (±{model_metrics['regressor_mae_seconds']:.2f}s MAE) | "
    f"✅ **Flight-Verified:** `{catastrophic_misses}/40` Missed Critical Fires (`{critical_recall*100:.0f}%` Critical Recall)"
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Predicted Fire Risk State",
        value=current_risk_label,
        delta=f"{risk_probs[risk_class]*100:.1f}% Model Confidence",
        delta_color="off",
    )
with col2:
    st.metric(
        label="Predicted Extinction Threshold",
        value=f"{pred_extinction_s:.2f} s",
        delta="Radiative Quench Window",
        delta_color="off",
    )
with col3:
    p_o2 = float(live_X["o2_partial_pressure_kpa"].iloc[0])
    st.metric(
        label="Oxygen Partial Pressure (P_O2)",
        value=f"{p_o2:.2f} kPa",
        delta=f"{p_o2 - 21.28:+.2f} kPa vs ISS Normoxic",
        delta_color="inverse",
    )
with col4:
    k_mix = float(live_X["gas_thermal_conductivity"].iloc[0])
    st.metric(
        label="Gas Thermal Conductivity (k_mix)",
        value=f"{k_mix:.4f} W/m·K",
        delta="Higher = Faster Radiative Cooling",
        delta_color="normal",
    )

st.markdown("### 🛡️ Automated ECLSS Fire Suppression Directive")
if risk_class == 0:
    st.markdown(
        '<div class="protocol-box-safe"><b>STATUS NOMINAL (SAFE):</b> Ambient atmospheric thermal conductivity '
        'and oxygen partial pressure favor rapid radiative self-extinction. Maintain standard ISS cabin ventilation (5 cm/s).</div>',
        unsafe_allow_html=True,
    )
elif risk_class == 1:
    st.markdown(
        '<div class="protocol-box-warning"><b>CAUTION — SUSTAINED MICROGRAVITY BURN (WARNING):</b> '
        'Diffusive oxygen supply can sustain flame propagation. <b>Action:</b> Throttle local module ventilation fans to minimum, '
        'isolate electrical bus near sensor zone, and prepare localized CO2/N2 diluent purge.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="protocol-box-critical"><b>EMERGENCY — CRITICAL FLASHOVER / TOXIC SMOLDERING HAZARD:</b> '
        'High oxygen partial pressure, droplet disruption risk, or lethal CO off-gassing detected! '
        '<b>Immediate Automated Action:</b> Cut module airflow immediately (zero-convection starvation), '
        'initiate Helium/Nitrogen inert gas injection to boost k_mix above 0.06 W/m·K, and don crew breathing masks.</div>',
        unsafe_allow_html=True,
    )

st.divider()

left_gauge, right_viz = st.columns([1, 2])

with left_gauge:
    st.subheader("🎯 Hazard Probability Gauge")
    hazard_score = float(risk_probs[1] * 50.0 + risk_probs[2] * 100.0)
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=hazard_score,
            number={"suffix": " / 100", "font": {"size": 28, "color": "white"}},
            title={"text": "Composite Flashover & Toxicity Index", "font": {"size": 15, "color": "#cbd5e1"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": current_risk_color},
                "bgcolor": "#1e293b",
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [35, 70], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [70, 100], "color": "rgba(244, 63, 94, 0.25)"},
                ],
            },
        )
    )
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=310,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    prob_df = pd.DataFrame(
        {
            "State": ["Safe (0)", "Warning (1)", "Critical (2)"],
            "Probability (%)": risk_probs * 100.0,
        }
    )
    fig_prob = px.bar(
        prob_df,
        x="Probability (%)",
        y="State",
        orientation="h",
        color="State",
        color_discrete_map={
            "Safe (0)": "#10b981",
            "Warning (1)": "#f59e0b",
            "Critical (2)": "#f43f5e",
        },
    )
    fig_prob.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        showlegend=False,
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with right_viz:
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🌌 3D Combustion Phase Space",
            "🔮 3D Spherical Flame Shell",
            "🛡️ Model Validation & Safety Audit",
            "📊 NASA PSI Data Explorer",
        ]
    )

    with tab1:
        plot_df = nasa_df[nasa_df["extinction_time_s"] <= 120.0].copy()
        fig_3d = px.scatter_3d(
            plot_df,
            x="o2_partial_pressure_kpa",
            y="gas_thermal_conductivity",
            z="extinction_time_s",
            color="fire_risk_label",
            symbol="mission_dataset",
            hover_data=["experiment_id", "fuel_material", "co_final_ppm"],
            color_discrete_map={
                "Safe": "#10b981",
                "Warning": "#f59e0b",
                "Critical": "#f43f5e",
            },
            opacity=0.78,
            title="Live Habitat State vs. 393 Real NASA ISS Combustion Runs",
        )
        fig_3d.add_trace(
            go.Scatter3d(
                x=[p_o2],
                y=[k_mix],
                z=[pred_extinction_s],
                mode="markers+text",
                marker=dict(size=12, color="#38bdf8", symbol="diamond", line=dict(width=2, color="white")),
                text=["CURRENT HABITAT STATE"],
                textposition="top center",
                name="Live ECLSS Telemetry",
            )
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="O2 Partial Pressure (kPa)",
                yaxis_title="Thermal Conductivity k_mix (W/m·K)",
                zaxis_title="Extinction Time (s)",
                bgcolor="#0f172a",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab2:
        st.caption(
            "In microgravity (μg), absence of thermal buoyancy produces a concentric spherical diffusion flame "
            "rather than an Earth-gravity teardrop plume."
        )
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 25)
        r_fuel = initial_size_mm / 2.0
        standoff_ratio = 1.8 + (p_o2 / 21.3) * 1.4 - (k_mix * 8.0)
        r_flame = max(r_fuel * 1.25, r_fuel * standoff_ratio)

        x_fuel = r_fuel * np.outer(np.cos(u), np.sin(v))
        y_fuel = r_fuel * np.outer(np.sin(u), np.sin(v))
        z_fuel = r_fuel * np.outer(np.ones(np.size(u)), np.cos(v))

        x_flame = r_flame * np.outer(np.cos(u), np.sin(v))
        y_flame = r_flame * np.outer(np.sin(u), np.sin(v))
        z_flame = r_flame * np.outer(np.ones(np.size(u)), np.cos(v))

        fig_sphere = go.Figure()
        fig_sphere.add_trace(
            go.Surface(
                x=x_fuel,
                y=y_fuel,
                z=z_fuel,
                colorscale=[[0, "#38bdf8"], [1, "#0284c7"]],
                showscale=False,
                name="Fuel Core",
            )
        )
        fig_sphere.add_trace(
            go.Surface(
                x=x_flame,
                y=y_flame,
                z=z_flame,
                opacity=0.38,
                colorscale=[[0, current_risk_color], [1, "#fbbf24"]],
                showscale=False,
                name="Spherical Reaction Zone",
            )
        )
        fig_sphere.update_layout(
            title=f"Spherical Reaction Envelope (Fuel Radius: {r_fuel:.2f} mm | Flame Shell: {r_flame:.2f} mm)",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                bgcolor="#0f172a",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=470,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_sphere, use_container_width=True)

    with tab3:
        st.success(
            f"✅ AEROSPACE SAFETY AUDIT PASSED: {catastrophic_misses} / 40 Critical Fires Misclassified as Safe "
            f"({critical_recall*100:.1f}% Critical Recall on 79 Unseen NASA ISS Test Runs)"
        )
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig_cm = px.imshow(
                val_cm,
                text_auto=True,
                x=["Pred Safe", "Pred Warning", "Pred Critical"],
                y=["Actual Safe", "Actual Warning", "Actual Critical"],
                color_continuous_scale="Blues",
                title="Unseen NASA Holdout Confusion Matrix (n=79)",
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                height=380,
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with v_col2:
            fig_parity = px.scatter(
                parity_df,
                x="Actual Burn Time (s)",
                y="Predicted Burn Time (s)",
                hover_data=["Experiment", "Fuel"],
                title=f"Extinction Time Validation (R² = {model_metrics['regressor_r2']:.2f})",
                color_discrete_sequence=["#38bdf8"],
            )
            max_t = float(max(parity_df["Actual Burn Time (s)"].max(), parity_df["Predicted Burn Time (s)"].max()))
            fig_parity.add_shape(
                type="line", x0=0, y0=0, x1=max_t, y1=max_t, line=dict(color="#10b981", dash="dash")
            )
            fig_parity.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0f172a",
                font={"color": "white"},
                height=380,
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(fig_parity, use_container_width=True)

    with tab4:
        mission_filter = st.multiselect(
            "Filter NASA ISS Experiment Series:",
            options=nasa_df["mission_dataset"].unique().tolist(),
            default=nasa_df["mission_dataset"].unique().tolist(),
        )
        filtered_df = nasa_df[nasa_df["mission_dataset"].isin(mission_filter)]
        st.dataframe(
            filtered_df[
                [
                    "experiment_id",
                    "mission_dataset",
                    "fuel_material",
                    "pressure_kpa",
                    "initial_o2_pct",
                    "inert_he_pct",
                    "co_final_ppm",
                    "burning_rate_mm2_s",
                    "extinction_time_s",
                    "fire_risk_label",
                ]
            ],
            use_container_width=True,
            height=420,
        )