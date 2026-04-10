import os

DATA_PATH="final_data_derived_with_flash_flood_indicator.csv"

ARTIFACT_DIR="artifacts"
MODEL_DIR=os.path.join(ARTIFACT_DIR,"models")

RANDOM_STATE=42
TEST_SIZE=0.3

FEATURE_NAMES=[
"Rain_Flag","Pressure_hPa","Temperature_K","Relative_Humidity",
"Wind_Speed_kmh","Wind_Direction_deg","Rain_Rate_mmph","LWP",
"Elevation_Angle","Azimuth_Angle","IWV","IRT1","RR_3hr","RR_6hr",
"RR_12hr","rain_intensity","dRR","Temp_C","dT","dP","P_3hr_drop",
"dWind","dLWP","dIWV","dIRT","dIWV_1h","dLWP_1h","dIRT_1h",
"dP_1h","dWind_1h"]