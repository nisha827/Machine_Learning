"""Train and compare reproducible Adult Income classification baselines."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .data import load_adult_data
from .pipeline import build_pipeline

CONFIGURATIONS = [
    ("logistic_regression", False),
    ("logistic_regression_balanced", False),
    ("random_forest", False),
    ("logistic_regression", True),
    ("random_forest", True),
]


def evaluate(y_true, predictions, probabilities) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def run(data_path: str | None = None, seed: int = 123, test_size: float = 0.25) -> pd.DataFrame:
    features, target = load_adult_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=seed, stratify=target
    )
    rows = []
    for model_name, resample in CONFIGURATIONS:
        estimator = build_pipeline(model_name, resample=resample, random_state=seed)
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        row = {"model": model_name, "random_oversampling": resample}
        row.update(evaluate(y_test, predictions, probabilities))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to the headerless Adult CSV (defaults to ./adult.csv)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--output", type=Path, help="Optional path to write metrics as CSV")
    args = parser.parse_args()
    results = run(args.data, args.seed, args.test_size)
    print(results.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
