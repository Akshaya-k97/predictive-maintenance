"""Data loading, schema validation and preprocessing.

Everything that touches the raw data lives here, so training and inference
share exactly the same logic (no duplicated preprocessing code).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"

RANDOM_SEED = 42   # one seed used everywhere for reproducibility
TEST_SIZE = 0.20

# Raw CSV column names (with spaces/units) -> clean names used in the code.
COLUMN_RENAME = {
    "Type": "Type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]          # never used as features
FAILURE_TYPE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_TYPE_NAMES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}
TARGET_COLUMN = "machine_failure"

CATEGORICAL_FEATURE = "Type"
TYPE_CATEGORIES = ["L", "M", "H"]                    # Low, Medium, High quality
NUMERIC_FEATURES = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
# NOTE: TWF/HDF/PWF/OSF/RNF are deliberately NOT features. They describe the
# failure itself, so using them would be target leakage.
FEATURE_COLUMNS = [CATEGORICAL_FEATURE] + NUMERIC_FEATURES

DISPLAY_NAMES = {
    "Type": "Machine type",
    "air_temp": "Air temperature",
    "process_temp": "Process temperature",
    "rpm": "Rotational speed",
    "torque": "Torque",
    "tool_wear": "Tool wear",
}
UNITS = {"air_temp": "K", "process_temp": "K", "rpm": "rpm", "torque": "Nm", "tool_wear": "min"}


class DataSchemaError(ValueError):
    """Raised when the CSV does not look like the AI4I 2020 dataset."""


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV, verify its schema, rename columns and drop identifiers."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download 'ai4i2020.csv' from the UCI repository "
            "(https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) "
            "and place it in the data/ folder. See data/README.md for details."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    expected = set(COLUMN_RENAME) | set(FAILURE_TYPE_COLUMNS)
    missing = expected - set(df.columns)
    if missing:
        raise DataSchemaError(
            f"The CSV is missing expected columns: {sorted(missing)}.\n"
            f"Columns found: {list(df.columns)}"
        )

    df = df.rename(columns=COLUMN_RENAME)
    df = df.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in df.columns])

    bad_types = set(df[CATEGORICAL_FEATURE].dropna().unique()) - set(TYPE_CATEGORIES)
    if bad_types:
        raise DataSchemaError(f"Unexpected values in 'Type': {bad_types}. Expected {TYPE_CATEGORIES}.")
    for col in NUMERIC_FEATURES + [TARGET_COLUMN] + FAILURE_TYPE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataSchemaError(f"Column '{col}' should be numeric but has dtype {df[col].dtype}.")
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summarise size, missing values, duplicates, class balance and outliers."""
    numeric = df[NUMERIC_FEATURES]
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    # IQR rule: points beyond 1.5 * IQR from the quartiles are *flagged*, not removed.
    outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()
    n_fail = int(df[TARGET_COLUMN].sum())
    return {
        "n_rows": int(len(df)),
        "n_features": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "missing_total": int(df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "n_duplicates": int(df.duplicated().sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "n_failures": n_fail,
        "n_no_failures": int(len(df) - n_fail),
        "failure_rate": float(n_fail / len(df)),
        "failure_type_counts": {c: int(df[c].sum()) for c in FAILURE_TYPE_COLUMNS},
        "outlier_counts_iqr": {k: int(v) for k, v in outliers.items()},
    }


def split_data(df: pd.DataFrame):
    """Stratified, reproducible train/test split. Done BEFORE any fitting."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)


def build_preprocessor() -> ColumnTransformer:
    """Unfitted preprocessor. It is fitted only inside a training pipeline.

    Output column order is [Type, air_temp, process_temp, rpm, torque, tool_wear]
    (identical to FEATURE_COLUMNS), so 'Type' is always column 0.
    """
    return ColumnTransformer(
        [
            ("type", OrdinalEncoder(categories=[TYPE_CATEGORIES], dtype=int), [CATEGORICAL_FEATURE]),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ]
    )
