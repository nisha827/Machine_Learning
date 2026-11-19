"""Data loading and schema validation for the UCI Adult Income dataset."""
from pathlib import Path

import pandas as pd

COLUMNS = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum", "MaritalStatus",
    "Occupation", "Relationship", "Race", "Gender", "CapitalGain", "CapitalLoss",
    "HoursPerWeek", "NativeCountry", "Income",
]
NUMERIC_FEATURES = ["Age", "fnlwgt", "EducationNum", "CapitalGain", "CapitalLoss", "HoursPerWeek"]
CATEGORICAL_FEATURES = [
    "WorkClass", "Education", "MaritalStatus", "Occupation", "Relationship",
    "Race", "Gender", "NativeCountry",
]
TARGET_MAP = {"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1}


def load_adult_data(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the headerless UCI Adult CSV shipped alongside this module's parent folder."""
    csv_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "adult.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Adult dataset not found at {csv_path}. Pass its path explicitly; "
            "the repository does not download data automatically."
        )
    frame = pd.read_csv(csv_path, header=None, names=COLUMNS, na_values="?", skipinitialspace=True)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, found {frame.shape[1]}")
    for column in CATEGORICAL_FEATURES + ["Income"]:
        frame[column] = frame[column].astype("string").str.strip()
    target = frame.pop("Income").map(TARGET_MAP)
    if target.isna().any():
        unknown = sorted(frame.loc[target.isna()].index[:5].tolist())
        raise ValueError(f"Unrecognised or missing income labels at row indices: {unknown}")
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, target.astype("int64")
