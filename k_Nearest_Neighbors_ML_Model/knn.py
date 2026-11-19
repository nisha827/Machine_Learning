"""Reusable K-nearest-neighbours models with train-fitted scaling.

The 2018 coursework scripts are retained as historical examples. This module is
the maintained implementation used by the accompanying tests.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_2d_float(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class MinMaxScaler:
    """Min-max scaler fitted only on training features."""

    def fit(self, values: ArrayLike) -> "MinMaxScaler":
        array = _as_2d_float(values, "values")
        self.minimum_ = array.min(axis=0)
        self.range_ = array.max(axis=0) - self.minimum_
        self.range_[self.range_ == 0.0] = 1.0
        return self

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "minimum_"):
            raise RuntimeError("Call fit before transform")
        array = _as_2d_float(values, "values")
        if array.shape[1] != self.minimum_.shape[0]:
            raise ValueError("Feature count differs from fitted data")
        return (array - self.minimum_) / self.range_

    def fit_transform(self, values: ArrayLike) -> NDArray[np.float64]:
        return self.fit(values).transform(values)


def pairwise_distances(
    training: ArrayLike, query: ArrayLike, metric: str = "euclidean", p: float = 2.0
) -> NDArray[np.float64]:
    train = _as_2d_float(training, "training")
    point = np.asarray(query, dtype=float)
    if point.ndim != 1 or point.shape[0] != train.shape[1] or not np.isfinite(point).all():
        raise ValueError("query must be a finite 1D vector matching the feature count")
    differences = np.abs(train - point)
    if metric == "euclidean":
        return np.sqrt(np.sum(differences ** 2, axis=1))
    if metric == "manhattan":
        return np.sum(differences, axis=1)
    if metric == "minkowski":
        if not np.isfinite(p) or p < 1:
            raise ValueError("Minkowski p must be finite and >= 1")
        return np.sum(differences ** p, axis=1) ** (1.0 / p)
    raise ValueError("metric must be 'euclidean', 'manhattan' or 'minkowski'")


class _KNNBase:
    def __init__(self, k: int = 3, metric: str = "euclidean", p: float = 2.0):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.metric = metric
        self.p = p

    def _fit_features(self, features: ArrayLike) -> NDArray[np.float64]:
        self.X_ = _as_2d_float(features, "features").copy()
        if self.k > len(self.X_):
            raise ValueError("k cannot exceed the number of training samples")
        return self.X_

    def _neighbors(self, query: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        distances = pairwise_distances(self.X_, query, self.metric, self.p)
        indices = np.argsort(distances, kind="stable")[:self.k]
        return indices, distances[indices]


class KNNClassifier(_KNNBase):
    def fit(self, features: ArrayLike, labels: Sequence[object]) -> "KNNClassifier":
        self._fit_features(features)
        self.y_ = np.asarray(labels)
        if self.y_.ndim != 1 or len(self.y_) != len(self.X_):
            raise ValueError("labels must be a 1D array matching the training samples")
        return self

    def predict(self, features: ArrayLike) -> NDArray:
        queries = _as_2d_float(features, "features")
        if queries.shape[1] != self.X_.shape[1]:
            raise ValueError("Feature count differs from fitted data")
        predictions = []
        for query in queries:
            indices, distances = self._neighbors(query)
            zero_mask = distances == 0.0
            selected = self.y_[indices[zero_mask]] if zero_mask.any() else self.y_[indices]
            counts = Counter(selected.tolist())
            predictions.append(sorted(counts, key=lambda label: (-counts[label], str(label)))[0])
        return np.asarray(predictions)


class KNNRegressor(_KNNBase):
    def fit(self, features: ArrayLike, targets: ArrayLike) -> "KNNRegressor":
        self._fit_features(features)
        self.y_ = np.asarray(targets, dtype=float)
        if self.y_.ndim != 1 or len(self.y_) != len(self.X_) or not np.isfinite(self.y_).all():
            raise ValueError("targets must be a finite 1D array matching the training samples")
        return self

    def predict(self, features: ArrayLike, weights: str = "uniform") -> NDArray[np.float64]:
        queries = _as_2d_float(features, "features")
        if queries.shape[1] != self.X_.shape[1]:
            raise ValueError("Feature count differs from fitted data")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        outputs = []
        for query in queries:
            indices, distances = self._neighbors(query)
            targets = self.y_[indices]
            zero_mask = distances == 0.0
            if zero_mask.any():
                outputs.append(float(np.mean(targets[zero_mask])))
            elif weights == "distance":
                inverse = 1.0 / distances
                outputs.append(float(np.average(targets, weights=inverse)))
            else:
                outputs.append(float(np.mean(targets)))
        return np.asarray(outputs)
