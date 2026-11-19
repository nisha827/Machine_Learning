import numpy as np
import pandas as pd

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_adult_data
from src.pipeline import build_pipeline, make_preprocessor


def sample_frame(n=40):
    frame = pd.DataFrame({
        "Age": np.arange(20, 20 + n, dtype=float),
        "fnlwgt": np.arange(100, 100 + n, dtype=float),
        "EducationNum": np.tile([9, 10, 13, 14], n // 4 + 1)[:n],
        "CapitalGain": np.zeros(n),
        "CapitalLoss": np.zeros(n),
        "HoursPerWeek": np.full(n, 40.0),
        "WorkClass": np.tile(["Private", "State-gov"], n // 2),
        "Education": np.tile(["HS-grad", "Bachelors"], n // 2),
        "MaritalStatus": np.tile(["Never-married", "Married-civ-spouse"], n // 2),
        "Occupation": np.tile(["Sales", "Tech-support"], n // 2),
        "Relationship": np.tile(["Not-in-family", "Husband"], n // 2),
        "Race": np.tile(["White", "Black"], n // 2),
        "Gender": np.tile(["Female", "Male"], n // 2),
        "NativeCountry": np.tile(["United-States", "India"], n // 2),
    })
    return frame


def test_preprocessor_keeps_numeric_and_categorical_features():
    frame = sample_frame()
    transformed = make_preprocessor().fit_transform(frame)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
    assert len(NUMERIC_FEATURES) == 6
    assert len(CATEGORICAL_FEATURES) == 8


def test_pipeline_fits_and_predicts_without_resampling():
    frame = sample_frame()
    target = pd.Series([0, 1] * 20)
    model = build_pipeline("logistic_regression")
    model.fit(frame.iloc[:32], target.iloc[:32])
    assert len(model.predict(frame.iloc[32:])) == 8


def test_resampler_is_inside_training_pipeline_only():
    frame = sample_frame()
    target = pd.Series([0] * 30 + [1] * 10)
    model = build_pipeline("logistic_regression", resample=True)
    model.fit(frame, target)
    assert model.named_steps["sampler"].sampling_strategy == "auto"
    assert len(model.predict(frame.iloc[:5])) == 5


def test_loader_reads_headerless_adult_schema_and_normalises_labels(tmp_path):
    row_a = ["39", " Private", "77516", " Bachelors", "13", " Never-married", " Adm-clerical", " Not-in-family", " White", " Male", "2174", "0", "40", " United-States", " <=50K"]
    row_b = ["50", " Self-emp", "83311", " Masters", "14", " Married-civ-spouse", " Exec-managerial", " Husband", " White", " Female", "0", "0", "50", " United-States", " >50K."]
    csv_path = tmp_path / "adult.csv"
    csv_path.write_text(",".join(row_a) + "\n" + ",".join(row_b) + "\n", encoding="utf-8")
    features, target = load_adult_data(csv_path)
    assert list(target) == [0, 1]
    assert features.loc[0, "WorkClass"] == "Private"
    assert features.loc[1, "Education"] == "Masters"
