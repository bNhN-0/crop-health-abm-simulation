from __future__ import annotations

import numpy as np
import pandas as pd


I_TERM_COLUMNS = [
    "i1_nutrient_level_balance",
    "i2_soil_context",
    "i3_atmospheric_support",
    "i4_atmospheric_stress",
    "i5_water_support",
    "i6_water_deficit",
    "i7_growth_readiness",
    "i8_biotic_env_stress",
    "i9_management_context",
]


def closeness(x: pd.Series, ref: pd.Series) -> pd.Series:
    return np.clip(1 - np.abs(x - ref), 0, 1)


def add_aggregated_features(df_prepared: pd.DataFrame) -> pd.DataFrame:
    df = df_prepared.copy()

    nutrient_quantity = 0.34 * df["n_scaled"] + 0.33 * df["p_scaled"] + 0.33 * df["k_scaled"]

    ratio_mean = (df["n_ratio_scaled"] + df["p_ratio_scaled"] + df["k_ratio_scaled"]) / 3.0
    ratio_dev = (
        np.abs(df["n_ratio_scaled"] - ratio_mean)
        + np.abs(df["p_ratio_scaled"] - ratio_mean)
        + np.abs(df["k_ratio_scaled"] - ratio_mean)
    ) / 3.0
    nutrient_balance = np.clip(1 - ratio_dev, 0, 1)

    df["i1_nutrient_level_balance"] = 0.60 * nutrient_quantity + 0.40 * nutrient_balance

    soil_fertility_base = (
        0.35 * df["ph_scaled"] + 0.40 * df["organic_matter_scaled"] + 0.25 * df["soil_nutrient_retention"]
    )
    fertility_match = closeness(soil_fertility_base, df["fertility_need"])
    df["i2_soil_context"] = 0.70 * soil_fertility_base + 0.30 * fertility_match

    temp_match = closeness(df["temperature_scaled"], df["opt_temp"])
    humidity_match = closeness(df["humidity_scaled"], df["opt_humidity"])
    sunlight_match = closeness(df["sunlight_exposure_scaled"], df["opt_sunlight"])
    df["i3_atmospheric_support"] = (
        0.30 * temp_match
        + 0.25 * humidity_match
        + 0.30 * sunlight_match
        + 0.15 * df["co2_concentration_scaled"]
    )

    df["i4_atmospheric_stress"] = (
        0.40 * df["thi_scaled"]
        + 0.40 * df["vpd_scaled"]
        + 0.10 * (1 - df["stress_tol"])
        + 0.10 * (1 - df["vpd_tolerance"])
    )

    water_supply = (
        0.40 * df["soil_moisture_scaled"]
        + 0.30 * df["rainfall_scaled"]
        + 0.20 * df["irrigation_frequency_scaled"]
        + 0.10 * (1 - df["water_usage_efficiency_scaled"])
    )
    water_context = (
        0.40 * df["soil_water_retention"]
        + 0.35 * df["water_reliability"]
        + 0.25 * df["water_quality"]
    )
    moisture_match = closeness(df["soil_moisture_scaled"], df["opt_moisture"])
    rainfall_match = closeness(df["rainfall_scaled"], df["opt_rainfall"])
    need_match = closeness(water_supply, df["water_need"])
    water_crop_match = 0.40 * moisture_match + 0.30 * rainfall_match + 0.30 * need_match
    df["i5_water_support"] = 0.40 * water_supply + 0.30 * water_context + 0.30 * water_crop_match

    df["i6_water_deficit"] = (
        0.55 * df["smd_scaled"]
        + 0.15 * (1 - df["soil_water_retention"])
        + 0.15 * (1 - df["water_reliability"])
        + 0.15 * df["water_need"]
    )

    df["i7_growth_readiness"] = (
        0.35 * df["gdd_scaled"]
        + 0.25 * df["growth_stage_norm"]
        + 0.25 * df["growth_rate"]
        + 0.15 * (1 - df["crop_density_scaled"])
    )

    df["i8_biotic_env_stress"] = 0.60 * df["pest_pressure_scaled"] + 0.40 * df["frost_risk_scaled"]
    df["i9_management_context"] = (
        0.50 * df["fertilizer_usage_scaled"] + 0.50 * df["urban_area_proximity_scaled"]
    )

    for column in I_TERM_COLUMNS:
        df[column] = np.clip(df[column], 0, 1)

    return df
