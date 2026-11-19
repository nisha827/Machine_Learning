"""Leakage-aware preprocessing and model pipeline construction."""
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_pipeline(model_name: str, resample: bool = False, random_state: int = 123) -> ImbPipeline:
    """Build preprocessing -> optional train-only random oversampling -> classifier."""
    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, class_weight=None, random_state=random_state)
    elif model_name == "logistic_regression_balanced":
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    steps = [("preprocess", make_preprocessor())]
    if resample:
        steps.append(("sampler", RandomOverSampler(random_state=random_state)))
    steps.append(("classifier", classifier))
    return ImbPipeline(steps)
