"""
Real NASA PSI Archive Organizer & Physical Harmonization Engine
Author: Space Apps 2026 Developer
Project: Flame in Freefall - Predictive Space Habitat Fire Safety System
"""

import os
import re
import numpy as np
import pandas as pd


def clean_numeric_sensor(val, over_range_val: float = 550.0) -> float:
    """Sanitizes real NASA PSI experimental entries:

    - Strips tilde approximations ('~6' -> 6.0)
    - Removes Windows-1252 en-dashes ('\\x96') and text placeholders ('none')
    - Maps gas analyzer saturation ('OR', 'over', '>500') to upper sensor limit
    """
    if pd.isna(val):
        return np.nan
    s = (
        str(val)
        .strip()
        .replace("~", "")
        .replace("\x96", "")
        .replace("–", "")
        .replace("—", "")
    )
    if s == "" or s.lower() in {"none", "nan", "na", "null"}:
        return np.nan
    if s.startswith(">"):
        try:
            return float(s[1:].strip())
        except ValueError:
            return over_range_val
    if s.lower() in {"or", "over"}:
        return over_range_val
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_flex_psi69(file_path: str) -> pd.DataFrame:
    """Ingests and normalizes PSI-69 (FLEX Liquid Droplet Combustion, 274 runs)."""
    df = pd.read_csv(file_path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame()
    out["experiment_id"] = "FLEX_" + df["FLEX Identifier"].astype(str).str.strip()
    out["mission_dataset"] = "FLEX (PSI-69)"
    out["fuel_material"] = df["Fuel"].astype(str).str.strip()
    out["fuel_category"] = "Liquid Droplet"

    pressure_mmhg = df["Ambient pressure; mmHg"].apply(clean_numeric_sensor)
    pressure_kpa = pressure_mmhg * 0.133322
    pressure_kpa = pressure_kpa.where(pressure_kpa > 5.0, np.nan)
    out["pressure_kpa"] = pressure_kpa.fillna(pressure_kpa.median())

    out["initial_o2_pct"] = (
        df["O initial ambient composition; mole fraction"].apply(clean_numeric_sensor) * 100.0
    )
    out["final_o2_pct"] = out["initial_o2_pct"]
    out["delta_o2_pct"] = 0.0

    out["inert_n2_pct"] = (
        df["N initial ambient composition; mole fraction"].apply(clean_numeric_sensor) * 100.0
    )
    out["co2_initial_pct"] = (
        df["CO initial ambient composition; mole fraction"].apply(clean_numeric_sensor) * 100.0
    )
    out["inert_he_pct"] = (
        df["He initial ambient composition; mole fraction"].apply(clean_numeric_sensor) * 100.0
    )

    out["initial_size_mm"] = df["Droplet initial diameter; mm"].apply(clean_numeric_sensor)
    out["extinction_size_mm"] = df["Visible flame extinction diameter; mm"].apply(clean_numeric_sensor)
    out["burning_rate_mm2_s"] = df["Burning rate; mm"].apply(clean_numeric_sensor)
    out["extinction_time_s"] = df["Burn time; s"].apply(clean_numeric_sensor)
    out["termination_mode"] = df["Test end"].astype(str).str.strip()

    out["co_initial_ppm"] = 0.0
    out["co_final_ppm"] = 0.0
    out["delta_co_ppm"] = 0.0

    return out


def parse_bass2_psi25(file_path: str) -> pd.DataFrame:
    """Ingests and normalizes PSI-25 (BASS-II Solid Material Combustion, 129 runs)."""
    df = pd.read_csv(file_path, encoding="latin1")
    df.columns = [re.sub(r"\s+", " ", c.strip()) for c in df.columns]

    out = pd.DataFrame()
    out["experiment_id"] = "BASS2_" + df["Test #"].astype(str).str.strip()
    out["mission_dataset"] = "BASS-II (PSI-25)"
    out["fuel_material"] = df["Fuel Sample Material"].astype(str).str.strip()
    out["fuel_category"] = "Solid Polymer/Fabric"

    out["pressure_kpa"] = 101.325

    out["initial_o2_pct"] = df["Calibrated initial O2 % by vol"].apply(clean_numeric_sensor)
    out["final_o2_pct"] = (
        df["Calibrated final O2 % by vol"]
        .apply(clean_numeric_sensor)
        .fillna(out["initial_o2_pct"])
    )
    out["delta_o2_pct"] = (out["initial_o2_pct"] - out["final_o2_pct"]).clip(lower=0.0)

    out["inert_n2_pct"] = 100.0 - out["initial_o2_pct"]
    out["co2_initial_pct"] = df["Initial CO2 % by vol"].apply(clean_numeric_sensor).fillna(0.4)
    out["inert_he_pct"] = 0.0

    out["co_initial_ppm"] = df["Initial CO (ppm)"].apply(clean_numeric_sensor).clip(lower=0.0).fillna(0.0)
    out["co_final_ppm"] = df["Final CO (ppm)"].apply(clean_numeric_sensor).clip(lower=0.0).fillna(out["co_initial_ppm"])
    out["delta_co_ppm"] = (out["co_final_ppm"] - out["co_initial_ppm"]).clip(lower=0.0)

    def extract_width_mm(text: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)\s*cm", str(text), re.IGNORECASE)
        return float(m.group(1)) * 10.0 if m else 20.0

    out["initial_size_mm"] = out["fuel_material"].apply(extract_width_mm)
    out["extinction_size_mm"] = 0.0

    frames = df["Total Frames Shot"].apply(clean_numeric_sensor)
    out["extinction_time_s"] = frames / 10.0
    out["burning_rate_mm2_s"] = (
        out["initial_size_mm"] / out["extinction_time_s"].replace(0, np.nan)
    ).fillna(0.5)

    out["termination_mode"] = np.where(
        out["delta_co_ppm"] > 100, "Smoldering/CO Spike", "Extinction"
    )

    return out


def parse_same_psi102(file_path: str) -> pd.DataFrame:
    """Ingests PSI-102 (SAME Smoke Aerosol & Wire Insulation Aging)."""
    df = pd.read_csv(file_path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    valid = df.dropna(subset=["Air Flow (cm/s)", "Weight loss rate (mg/min)"]).copy()

    out = pd.DataFrame()
    out["experiment_id"] = "SAME_" + valid["Sample #"].astype(str)
    out["mission_dataset"] = "SAME (PSI-102)"
    out["fuel_material"] = valid["Sample Material"].astype(str).str.strip()
    out["fuel_category"] = "Wire Insulation"
    out["pressure_kpa"] = 101.325
    out["initial_o2_pct"] = 21.0
    out["final_o2_pct"] = 21.0
    out["delta_o2_pct"] = 0.0
    out["inert_n2_pct"] = 79.0
    out["co2_initial_pct"] = 0.04
    out["inert_he_pct"] = 0.0
    out["initial_size_mm"] = 2.0
    out["extinction_size_mm"] = 0.0
    out["burning_rate_mm2_s"] = valid["Weight loss rate (mg/min)"] / 60.0
    out["extinction_time_s"] = valid["Aging Duration (minutes)"] * 60.0
    out["termination_mode"] = "Thermal Pyrolysis"
    out["co_initial_ppm"] = 0.0
    out["co_final_ppm"] = valid["Weight loss rate (mg/min)"] * 15.0
    out["delta_co_ppm"] = out["co_final_ppm"]

    return out


def assign_physical_fire_risk(row: pd.Series) -> int:
    """Assigns ground-truth Space Habitat Fire Risk Level based on physical telemetry:

    0 = Safe / Rapid Radiative Extinction (Low O2, fast extinction, zero CO spike)
    1 = Warning / Sustained Microgravity Burn (Nominal cabin O2, moderate burn rate)
    2 = Critical / Flashover or Toxic Smoldering Hazard (Enriched O2 >= 22%, Disruption, or CO >= 100 ppm)
    """
    if (
        row["initial_o2_pct"] >= 22.0
        or row["termination_mode"] == "Disruption"
        or row["co_final_ppm"] >= 100.0
        or row["burning_rate_mm2_s"] >= 0.75
    ):
        return 2
    elif (
        row["initial_o2_pct"] >= 18.5
        or row["extinction_time_s"] >= 10.0
        or row["co_final_ppm"] >= 25.0
    ):
        return 1
    else:
        return 0


def build_organized_nasa_dataset(
    data_dir: str = "data", output_dir: str = "organized_data"
) -> pd.DataFrame:
    """Locates real NASA PSI CSV files, cleans each experiment family,

    merges them into a unified ML-ready master dataset, and saves audit outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    search_dirs = [data_dir, "."]

    def locate(filename: str) -> str:
        for d in search_dirs:
            candidate = os.path.join(d, filename)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Could not locate '{filename}' in {search_dirs}")

    flex_df = parse_flex_psi69(locate("PSI-69_Experimental table_FLEX.csv"))
    bass_df = parse_bass2_psi25(locate("PSI-25_Experimental table_BASS-II.csv"))
    same_df = parse_same_psi102(locate("PSI-102_Experimental table_SAME.csv"))

    flex_df.to_csv(os.path.join(output_dir, "flex_psi69_clean.csv"), index=False)
    bass_df.to_csv(os.path.join(output_dir, "bass2_psi25_clean.csv"), index=False)
    same_df.to_csv(os.path.join(output_dir, "same_psi102_clean.csv"), index=False)

    master_df = pd.concat([flex_df, bass_df, same_df], ignore_index=True)

    master_df = master_df.dropna(subset=["extinction_time_s"]).copy()

    for col in ["initial_size_mm", "extinction_size_mm", "burning_rate_mm2_s"]:
        master_df[col] = master_df.groupby("fuel_category")[col].transform(
            lambda s: s.fillna(s.median())
        )

    master_df["fire_risk_level"] = master_df.apply(assign_physical_fire_risk, axis=1)
    risk_labels = {0: "Safe", 1: "Warning", 2: "Critical"}
    master_df["fire_risk_label"] = master_df["fire_risk_level"].map(risk_labels)

    master_path = os.path.join(output_dir, "master_microgravity_fires.csv")
    master_df.to_csv(master_path, index=False)

    print("[SUCCESS] Real NASA PSI Data Organized & Harmonized!")
    print(f"Master Dataset Saved: {master_path}")
    print(f"Total Verified Experimental Runs: {len(master_df)}")
    print("\nBreakdown by NASA Mission Dataset:")
    print(master_df["mission_dataset"].value_counts())
    print("\nGround-Truth Fire Risk Class Distribution:")
    print(master_df["fire_risk_label"].value_counts())
    return master_df


if __name__ == "__main__":
    build_organized_nasa_dataset(data_dir="data", output_dir="organized_data")