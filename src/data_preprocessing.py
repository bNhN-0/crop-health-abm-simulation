from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


REQUIRED_COLUMNS = [
    "n",
    "p",
    "k",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "label",
    "soil_moisture",
    "soil_type",
    "sunlight_exposure",
    "wind_speed",
    "co2_concentration",
    "organic_matter",
    "irrigation_frequency",
    "crop_density",
    "pest_pressure",
    "fertilizer_usage",
    "growth_stage",
    "urban_area_proximity",
    "water_source_type",
    "frost_risk",
    "water_usage_efficiency",
]

SCALE_COLUMNS = [
    "n",
    "p",
    "k",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "soil_moisture",
    "sunlight_exposure",
    "wind_speed",
    "co2_concentration",
    "organic_matter",
    "irrigation_frequency",
    "crop_density",
    "pest_pressure",
    "fertilizer_usage",
    "urban_area_proximity",
    "frost_risk",
    "water_usage_efficiency",
]

DERIVED_COLUMNS = [
    "thi",
    "vpd",
    "gdd",
    "smd",
    "n_ratio",
    "p_ratio",
    "k_ratio",
]

SOIL_PARAMS = {
    1: {"water_retention": 0.40, "nutrient_retention": 0.50},
    2: {"water_retention": 0.70, "nutrient_retention": 0.70},
    3: {"water_retention": 0.90, "nutrient_retention": 0.80},
}

WATER_SOURCE_PARAMS = {
    1: {"reliability": 0.85, "quality": 0.80},
    2: {"reliability": 0.75, "quality": 0.90},
    3: {"reliability": 0.65, "quality": 0.70},
}

FIELD_CAPACITY = {
    1: 0.40,
    2: 0.70,
    3: 0.90,
}


@dataclass
class PreprocessingArtifacts:
    prepared_df: pd.DataFrame
    crop_params_df: pd.DataFrame
    full_feature_df: pd.DataFrame


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(r"[()/%Â²Â°]", "", regex=True)
        .str.replace(r"[/\-]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )

    rename_map = {
        "n_ppm": "n",
        "p_ppm": "p",
        "k_ppm": "k",
        "temperature_c": "temperature",
        "rainfall_mm": "rainfall",
        "sunlight_exposure_hrs_day": "sunlight_exposure",
        "wind_speed_kmh": "wind_speed",
        "co2_concentration_ppm": "co2_concentration",
        "irrigation_frequency_times_week": "irrigation_frequency",
        "crop_density_plants_m": "crop_density",
        "fertilizer_usage_kg_ha": "fertilizer_usage",
        "urban_area_proximity_km": "urban_area_proximity",
        "water_usage_efficiency_l_kg": "water_usage_efficiency",
    }
    return cleaned.rename(columns={k: v for k, v in rename_map.items() if k in cleaned.columns})


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_required = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    numeric_cols = prepared.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = prepared.select_dtypes(include=["object"]).columns.tolist()

    for column in numeric_cols:
        prepared[column] = prepared[column].fillna(prepared[column].median())

    for column in object_cols:
        if prepared[column].isna().sum() > 0:
            prepared[column] = prepared[column].fillna(prepared[column].mode()[0])

    return prepared


def normalize_discrete_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in ["soil_type", "water_source_type", "growth_stage"]:
        prepared[column] = prepared[column].round().astype(int).clip(1, 3)
    return prepared


def add_scaled_base_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    scaler = MinMaxScaler()
    scaled_base_names = [f"{column}_scaled" for column in SCALE_COLUMNS]
    prepared[scaled_base_names] = scaler.fit_transform(prepared[SCALE_COLUMNS])
    return prepared


def add_derived_features(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    prepared = df.copy()

    prepared["thi"] = prepared["temperature"] - (
        (0.55 - 0.0055 * prepared["humidity"]) * (prepared["temperature"] - 14.5)
    )

    es = 0.6108 * np.exp((17.27 * prepared["temperature"]) / (prepared["temperature"] + 237.3))
    prepared["vpd"] = es * (1 - prepared["humidity"] / 100)

    t_base = 10
    prepared["gdd"] = np.maximum(0, prepared["temperature"] - t_base)

    prepared["fc"] = prepared["soil_type"].map(FIELD_CAPACITY)
    prepared["smd"] = prepared["fc"] - prepared["soil_moisture"]

    total_npk = prepared["n"] + prepared["p"] + prepared["k"] + eps
    prepared["n_ratio"] = prepared["n"] / total_npk
    prepared["p_ratio"] = prepared["p"] / total_npk
    prepared["k_ratio"] = prepared["k"] / total_npk

    return prepared


def add_scaled_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    scaler = MinMaxScaler()
    derived_scaled_names = [f"{column}_scaled" for column in DERIVED_COLUMNS]
    prepared[derived_scaled_names] = scaler.fit_transform(prepared[DERIVED_COLUMNS])
    return prepared


def build_crop_params(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    crop_params: dict[str, dict[str, float]] = {}

    for crop, group in df.groupby("label"):
        crop_params[crop] = {
            "opt_temp": group["temperature_scaled"].mean(),
            "opt_humidity": group["humidity_scaled"].mean(),
            "opt_moisture": group["soil_moisture_scaled"].mean(),
            "opt_rainfall": group["rainfall_scaled"].mean(),
            "opt_ph": group["ph_scaled"].mean(),
            "opt_sunlight": group["sunlight_exposure_scaled"].mean(),
            "stress_tol": 1 - group["frost_risk_scaled"].mean(),
            "growth_rate": group["gdd_scaled"].mean(),
            "water_need": group["soil_moisture_scaled"].mean(),
            "vpd_tolerance": 1 - group["vpd_scaled"].mean(),
            "fertility_need": (
                0.30 * group["n_scaled"].mean()
                + 0.25 * group["p_scaled"].mean()
                + 0.25 * group["k_scaled"].mean()
                + 0.10 * group["organic_matter_scaled"].mean()
                + 0.10 * group["ph_scaled"].mean()
            ),
        }

    return crop_params


def add_parameter_switch_features(
    df: pd.DataFrame,
    crop_params: dict[str, dict[str, float]],
) -> pd.DataFrame:
    prepared = df.copy()

    prepared["growth_stage_norm"] = prepared["growth_stage"] / prepared["growth_stage"].max()
    prepared["soil_water_retention"] = prepared["soil_type"].map(
        {key: value["water_retention"] for key, value in SOIL_PARAMS.items()}
    )
    prepared["soil_nutrient_retention"] = prepared["soil_type"].map(
        {key: value["nutrient_retention"] for key, value in SOIL_PARAMS.items()}
    )
    prepared["water_reliability"] = prepared["water_source_type"].map(
        {key: value["reliability"] for key, value in WATER_SOURCE_PARAMS.items()}
    )
    prepared["water_quality"] = prepared["water_source_type"].map(
        {key: value["quality"] for key, value in WATER_SOURCE_PARAMS.items()}
    )

    crop_param_keys = list(next(iter(crop_params.values())).keys())
    for key in crop_param_keys:
        prepared[key] = prepared["label"].map({label: params[key] for label, params in crop_params.items()})

    return prepared


def select_prepared_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared_columns = [
        "label",
        "soil_type",
        "water_source_type",
        "growth_stage",
        "n_scaled",
        "p_scaled",
        "k_scaled",
        "temperature_scaled",
        "humidity_scaled",
        "ph_scaled",
        "rainfall_scaled",
        "soil_moisture_scaled",
        "sunlight_exposure_scaled",
        "wind_speed_scaled",
        "co2_concentration_scaled",
        "organic_matter_scaled",
        "irrigation_frequency_scaled",
        "crop_density_scaled",
        "pest_pressure_scaled",
        "fertilizer_usage_scaled",
        "urban_area_proximity_scaled",
        "frost_risk_scaled",
        "water_usage_efficiency_scaled",
        "thi_scaled",
        "vpd_scaled",
        "gdd_scaled",
        "smd_scaled",
        "n_ratio_scaled",
        "p_ratio_scaled",
        "k_ratio_scaled",
        "growth_stage_norm",
        "soil_water_retention",
        "soil_nutrient_retention",
        "water_reliability",
        "water_quality",
        "opt_temp",
        "opt_humidity",
        "opt_moisture",
        "opt_rainfall",
        "opt_ph",
        "opt_sunlight",
        "stress_tol",
        "growth_rate",
        "water_need",
        "vpd_tolerance",
        "fertility_need",
    ]
    return df[prepared_columns].copy()


def preprocess_dataset(df: pd.DataFrame) -> PreprocessingArtifacts:
    cleaned = clean_column_names(df)
    validate_required_columns(cleaned)
    prepared = fill_missing_values(cleaned)
    prepared = normalize_discrete_columns(prepared)
    prepared = add_scaled_base_features(prepared)
    prepared = add_derived_features(prepared)
    prepared = add_scaled_derived_features(prepared)

    crop_params = build_crop_params(prepared)
    prepared = add_parameter_switch_features(prepared, crop_params)

    prepared_df = select_prepared_columns(prepared)
    crop_params_df = pd.DataFrame(crop_params).T.reset_index().rename(columns={"index": "label"})

    return PreprocessingArtifacts(
        prepared_df=prepared_df,
        crop_params_df=crop_params_df,
        full_feature_df=prepared,
    )


def preprocess_csv(input_path: str | Path) -> PreprocessingArtifacts:
    df = load_dataset(input_path)
    return preprocess_dataset(df)
