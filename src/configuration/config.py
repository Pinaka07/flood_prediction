import os

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH    = "final_data_derived_with_flash_flood_indicator.csv"
ARTIFACT_DIR = "artifacts"
MODEL_DIR    = os.path.join(ARTIFACT_DIR, "models")
PLOT_DIR     = os.path.join(ARTIFACT_DIR, "plots")

# ─── Split / reproducibility ───────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.30

# ─── Class imbalance ───────────────────────────────────────────────────────
# Dataset: 143 360 No-Flood vs 5 126 Flood  →  ratio ≈ 28 : 1
# SMOTE grows minority to 20 % of majority (safer than forcing 50/50).
SMOTE_SAMPLING_STRATEGY = 0.20

# ─── Probability threshold ─────────────────────────────────────────────────
# Default 0.5 is wrong for heavily imbalanced data.
# The pipeline sweeps [0.01, 0.99] to find the F1-optimal threshold per model.
PROB_THRESHOLD_DEFAULT = 0.5   # fallback when y_prob is unavailable

# ─── Leakage audit ─────────────────────────────────────────────────────────
# Features whose |Pearson r| with target exceeds this value are dropped.
LEAKAGE_CORR_THRESHOLD = 0.80

# Cumulative rain aggregates that almost directly encode the flood label.
LEAKAGE_SUSPECTS = ["RR_3hr", "RR_6hr", "RR_12hr", "rain_intensity", "dRR"]

# ─── Features ──────────────────────────────────────────────────────────────
ALL_FEATURE_NAMES = [
    "Rain_Flag", "Pressure_hPa", "Temperature_K", "Relative_Humidity",
    "Wind_Speed_kmh", "Wind_Direction_deg", "Rain_Rate_mmph", "LWP",
    "Elevation_Angle", "Azimuth_Angle", "IWV", "IRT1", "RR_3hr", "RR_6hr",
    "RR_12hr", "rain_intensity", "dRR", "Temp_C", "dT", "dP", "P_3hr_drop",
    "dWind", "dLWP", "dIWV", "dIRT", "dIWV_1h", "dLWP_1h", "dIRT_1h",
    "dP_1h", "dWind_1h",
]

# Safe feature list — leakage suspects removed at config level.
# data_ingestion.leakage_audit() may remove additional columns at runtime
# and update this list module-wide.
FEATURE_NAMES = [f for f in ALL_FEATURE_NAMES if f not in LEAKAGE_SUSPECTS]
