from src.utils.main_utils import read_yaml

def validate_data(df):
    schema=read_yaml("config/schema.yaml")

    expected_cols=list(schema["columns"].keys())

    missing_cols=[col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise Exception(f"Missing columns: {missing_cols}")

    if schema["target_column"] not in df.columns:
        raise Exception("Target column missing")

    return True