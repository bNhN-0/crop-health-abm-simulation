from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_T = 800
DEFAULT_DT = 0.01
DEFAULT_ETA_GO = 0.7
DEFAULT_ETA_IR = 0.9
DEFAULT_ETA_CV = 0.5

SIGMA1 = 0.5
SIGMA2 = 0.5
ALPHA2 = 0.75
ALPHA3 = 0.6
BETA3 = 0.5
DELTA4 = 0.6667
ALPHA4 = 0.3333
BETA4 = 0.6667
LAMBDA4 = 0.5
ALPHA5 = 0.3
BETA5 = 0.8571
LAMBDA5 = 1.0
ALPHA6 = 0.8571
ALPHA7 = 0.6
BETA7 = 0.5
ALPHA8 = 0.5
ALPHA9 = 0.5
DELTA10 = 0.6
ALPHA10 = 0.6
BETA10 = 0.5
ALPHA11 = 0.5
ALPHA12 = 0.6
BETA12 = 0.8


OUTPUT_COLUMNS = [
    "Sc (Soil_Condition)",
    "Ac (Atmospheric_Condition)",
    "Wc (Water_Condition)",
    "Gc (Growth_Condition)",
    "Cs (Contextual_Sensitivity)",
    "Es (Environmental_Suitability)",
    "Gp (Growth_Potential)",
    "Sp (Stress_Pressure)",
    "Rp (Risk_Propagation)",
    "Sb (Stability)",
    "Go (Growth_Outlook)",
    "Ir (Immediate_Risk)",
    "Cv (Crop_Viability)",
    "LGo (LongTerm_Growth_Outlook)",
    "LIr (LongTerm_Immediate_Risk)",
    "LCv (LongTerm_Crop_Viability)",
]


@dataclass
class SimulationConfig:
    T: int = DEFAULT_T
    dt: float = DEFAULT_DT
    eta_go: float = DEFAULT_ETA_GO
    eta_ir: float = DEFAULT_ETA_IR
    eta_cv: float = DEFAULT_ETA_CV


def clip(value: np.ndarray | float) -> np.ndarray | float:
    return np.clip(value, 0.05, 0.95)


def classify_support(value: float) -> str:
    if value >= 0.75:
        return "High Support"
    if value >= 0.55:
        return "Moderate Support"
    if value >= 0.35:
        return "Low Support"
    return "Very Low Support"


def classify_disturbance(value: float) -> str:
    if value <= 0.25:
        return "Low Disturbance"
    if value <= 0.45:
        return "Mild Disturbance"
    if value <= 0.65:
        return "High Disturbance"
    return "Severe Disturbance"


def classify_crop_viability(value: float) -> str:
    if value >= 0.75:
        return "Healthy"
    if value >= 0.55:
        return "Moderate"
    if value >= 0.35:
        return "Warning"
    return "Critical"


def run_crop_model(row: pd.Series, config: SimulationConfig | None = None) -> dict[str, np.ndarray]:
    config = config or SimulationConfig()
    T = config.T

    i1 = np.ones(T) * row["i1_nutrient_level_balance"]
    i2 = np.ones(T) * row["i2_soil_context"]
    i3 = np.ones(T) * row["i3_atmospheric_support"]
    i4 = np.ones(T) * row["i4_atmospheric_stress"]
    i5 = np.ones(T) * row["i5_water_support"]
    i6 = np.ones(T) * row["i6_water_deficit"]
    i7 = np.ones(T) * row["i7_growth_readiness"]
    i8 = np.ones(T) * row["i8_biotic_env_stress"]
    i9 = np.ones(T) * row["i9_management_context"]

    Sc = np.zeros(T)
    Ac = np.zeros(T)
    Wc = np.zeros(T)
    Gc = np.zeros(T)
    Cs = np.zeros(T)
    Es = np.zeros(T)
    Gp = np.zeros(T)
    Sp = np.zeros(T)
    Rp = np.zeros(T)
    Sb = np.zeros(T)
    Go = np.zeros(T)
    Ir = np.zeros(T)
    Cv = np.zeros(T)
    LGo = np.zeros(T)
    LIr = np.zeros(T)
    LCv = np.zeros(T)

    Sc[0] = clip(SIGMA1 * i1[0] + SIGMA2 * i2[0])
    Ac[0] = clip((1 - i4[0]) * (ALPHA2 * i3[0] + (1 - ALPHA2) * i7[0]))
    Wc[0] = clip((1 - i5[0]) * (ALPHA3 * i6[0] + (1 - ALPHA3) * (BETA3 * (i4[0] + i7[0]))))
    Gc[0] = clip(
        (1 - (DELTA4 * i4[0] + (1 - DELTA4) * i6[0]))
        * (
            ALPHA4 * i7[0]
            + (1 - ALPHA4) * (BETA4 * (i5[0] + i3[0]) + (1 - BETA4) * (LAMBDA4 * (i1[0] + i2[0])))
        )
    )
    Cs[0] = clip(
        ALPHA5 * i8[0]
        + (1 - ALPHA5) * (BETA5 * (i4[0] + i6[0] + i9[0]) + (1 - BETA5) * (LAMBDA5 * i7[0]))
    )
    Es[0] = clip(ALPHA6 * (Sc[0] + Ac[0]) + (1 - ALPHA6) * Gc[0])
    Gp[0] = clip((1 - Wc[0]) * (ALPHA7 * Gc[0] + (1 - ALPHA7) * (BETA7 * (Sc[0] + Ac[0]))))
    Sp[0] = clip((1 - Ac[0]) * (ALPHA8 * (Wc[0] + Cs[0])))
    Rp[0] = clip((1 - Gc[0]) * (ALPHA9 * (Wc[0] + Cs[0])))
    Sb[0] = clip(
        (1 - (DELTA10 * Wc[0] + (1 - DELTA10) * Cs[0]))
        * (ALPHA10 * Sc[0] + (1 - ALPHA10) * (BETA10 * (Ac[0] + Gc[0])))
    )
    Go[0] = clip((1 - Sp[0]) * (ALPHA12 * (Es[0] + Gp[0]) + (1 - ALPHA12) * (BETA12 * Sb[0])))
    Ir[0] = clip((1 - Sb[0]) * (ALPHA11 * (Sp[0] + Rp[0])))
    Cv[0] = clip(Go[0] * (1 - Ir[0]))

    LGo[0] = 0.1
    LIr[0] = 0.9
    LCv[0] = 0.1

    for t in range(1, T):
        Sc[t] = clip(SIGMA1 * i1[t] + SIGMA2 * i2[t])
        Ac[t] = clip((1 - i4[t]) * (ALPHA2 * i3[t] + (1 - ALPHA2) * i7[t]))
        Wc[t] = clip((1 - i5[t]) * (ALPHA3 * i6[t] + (1 - ALPHA3) * (BETA3 * (i4[t] + i7[t]))))
        Gc[t] = clip(
            (1 - (DELTA4 * i4[t] + (1 - DELTA4) * i6[t]))
            * (
                ALPHA4 * i7[t]
                + (1 - ALPHA4) * (BETA4 * (i5[t] + i3[t]) + (1 - BETA4) * (LAMBDA4 * (i1[t] + i2[t])))
            )
        )
        Cs[t] = clip(
            ALPHA5 * i8[t]
            + (1 - ALPHA5) * (BETA5 * (i4[t] + i6[t] + i9[t]) + (1 - BETA5) * (LAMBDA5 * i7[t]))
        )
        Es[t] = clip(ALPHA6 * (Sc[t] + Ac[t]) + (1 - ALPHA6) * Gc[t])
        Gp[t] = clip((1 - Wc[t]) * (ALPHA7 * Gc[t] + (1 - ALPHA7) * (BETA7 * (Sc[t] + Ac[t]))))
        Sp[t] = clip((1 - Ac[t]) * (ALPHA8 * (Wc[t] + Cs[t])))
        Rp[t] = clip((1 - Gc[t]) * (ALPHA9 * (Wc[t] + Cs[t])))
        Sb[t] = clip(
            (1 - (DELTA10 * Wc[t] + (1 - DELTA10) * Cs[t]))
            * (ALPHA10 * Sc[t] + (1 - ALPHA10) * (BETA10 * (Ac[t] + Gc[t])))
        )
        Go[t] = clip((1 - Sp[t]) * (ALPHA12 * (Es[t] + Gp[t]) + (1 - ALPHA12) * (BETA12 * Sb[t])))
        Ir[t] = clip((1 - Sb[t]) * (ALPHA11 * (Sp[t] + Rp[t])))
        Cv[t] = clip(Go[t] * (1 - Ir[t]))

        LGo[t] = clip(LGo[t - 1] + config.eta_go * (Go[t - 1] - LGo[t - 1]) * config.dt)
        LIr[t] = clip(LIr[t - 1] + config.eta_ir * (Ir[t - 1] - LIr[t - 1]) * config.dt)
        LCv[t] = clip(LCv[t - 1] + config.eta_cv * (Cv[t - 1] - LCv[t - 1]) * config.dt)

    return {
        "Sc (Soil_Condition)": Sc,
        "Ac (Atmospheric_Condition)": Ac,
        "Wc (Water_Condition)": Wc,
        "Gc (Growth_Condition)": Gc,
        "Cs (Contextual_Sensitivity)": Cs,
        "Es (Environmental_Suitability)": Es,
        "Gp (Growth_Potential)": Gp,
        "Sp (Stress_Pressure)": Sp,
        "Rp (Risk_Propagation)": Rp,
        "Sb (Stability)": Sb,
        "Go (Growth_Outlook)": Go,
        "Ir (Immediate_Risk)": Ir,
        "Cv (Crop_Viability)": Cv,
        "LGo (LongTerm_Growth_Outlook)": LGo,
        "LIr (LongTerm_Immediate_Risk)": LIr,
        "LCv (LongTerm_Crop_Viability)": LCv,
    }


def simulate_dataset(df: pd.DataFrame, config: SimulationConfig | None = None) -> tuple[pd.DataFrame, list[dict[str, np.ndarray]]]:
    config = config or SimulationConfig()
    results = []
    trajectories: list[dict[str, np.ndarray]] = []

    for _, row in df.iterrows():
        simulation = run_crop_model(row, config=config)
        trajectories.append(simulation)
        results.append(
            {
                "crop_type": row["label"],
                "Sc (Soil_Condition)": round(simulation["Sc (Soil_Condition)"][-1], 4),
                "Ac (Atmospheric_Condition)": round(simulation["Ac (Atmospheric_Condition)"][-1], 4),
                "Wc (Water_Condition)": round(simulation["Wc (Water_Condition)"][-1], 4),
                "Gc (Growth_Condition)": round(simulation["Gc (Growth_Condition)"][-1], 4),
                "Cs (Contextual_Sensitivity)": round(simulation["Cs (Contextual_Sensitivity)"][-1], 4),
                "Es (Environmental_Suitability)": round(simulation["Es (Environmental_Suitability)"][-1], 4),
                "Gp (Growth_Potential)": round(simulation["Gp (Growth_Potential)"][-1], 4),
                "Sp (Stress_Pressure)": round(simulation["Sp (Stress_Pressure)"][-1], 4),
                "Rp (Risk_Propagation)": round(simulation["Rp (Risk_Propagation)"][-1], 4),
                "Sb (Stability)": round(simulation["Sb (Stability)"][-1], 4),
                "Go (Growth_Outlook)": round(simulation["Go (Growth_Outlook)"][-1], 4),
                "Ir (Immediate_Risk)": round(simulation["Ir (Immediate_Risk)"][-1], 4),
                "Cv (Crop_Viability)": round(simulation["Cv (Crop_Viability)"][-1], 4),
                "LGo (LongTerm_Growth_Outlook)": round(simulation["LGo (LongTerm_Growth_Outlook)"][-1], 4),
                "LIr (LongTerm_Immediate_Risk)": round(simulation["LIr (LongTerm_Immediate_Risk)"][-1], 4),
                "LCv (LongTerm_Crop_Viability)": round(simulation["LCv (LongTerm_Crop_Viability)"][-1], 4),
                "support_state": classify_support(simulation["LGo (LongTerm_Growth_Outlook)"][-1]),
                "disturbance_state": classify_disturbance(simulation["LIr (LongTerm_Immediate_Risk)"][-1]),
                "crop_state": classify_crop_viability(simulation["LCv (LongTerm_Crop_Viability)"][-1]),
            }
        )

    return pd.DataFrame(results), trajectories


def summarize_results(results_df: pd.DataFrame) -> dict[str, pd.DataFrame | float | str]:
    average_viability = results_df["LCv (LongTerm_Crop_Viability)"].mean()
    state_counts = results_df["crop_state"].value_counts()
    state_percentages = results_df["crop_state"].value_counts(normalize=True) * 100

    crop_health_summary = (
        results_df.groupby("crop_type")["LCv (LongTerm_Crop_Viability)"]
        .mean()
        .reset_index(name="avg_health")
        .sort_values(by="avg_health", ascending=False)
    )
    crop_health_summary["overall_state"] = crop_health_summary["avg_health"].apply(classify_crop_viability)

    summary_table = pd.DataFrame({"count": state_counts, "percentage": state_percentages.round(2)})

    return {
        "average_viability": float(average_viability),
        "overall_state": classify_crop_viability(float(average_viability)),
        "state_counts": state_counts,
        "state_percentages": state_percentages.round(2),
        "summary_table": summary_table,
        "crop_health_summary": crop_health_summary,
    }
