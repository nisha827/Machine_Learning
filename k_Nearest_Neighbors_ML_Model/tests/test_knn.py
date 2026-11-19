import numpy as np
import pytest

from k_Nearest_Neighbors_ML_Model.knn import (
    KNNClassifier, KNNRegressor, MinMaxScaler, pairwise_distances,
)


def test_distance_metrics():
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    q = np.array([0.0, 0.0])
    assert pairwise_distances(x, q, "euclidean").tolist() == pytest.approx([0.0, 5.0])
    assert pairwise_distances(x, q, "manhattan").tolist() == pytest.approx([0.0, 7.0])
    assert pairwise_distances(x, q, "minkowski", p=3).tolist() == pytest.approx([0.0, 4.497941445])


def test_scaler_uses_training_statistics_and_handles_constant_column():
    scaler = MinMaxScaler().fit([[0, 5], [10, 5]])
    transformed = scaler.transform([[20, 5]])
    assert transformed.tolist() == [[2.0, 0.0]]


def test_classifier_prediction_and_zero_distance():
    model = KNNClassifier(k=3).fit([[0], [1], [2], [3]], ["a", "a", "b", "b"])
    assert model.predict([[0]])[0] == "a"


def test_regression_uniform_and_distance_weights():
    model = KNNRegressor(k=2).fit([[0], [2], [4]], [0, 2, 4])
    assert model.predict([[1]])[0] == pytest.approx(1.0)
    assert model.predict([[0]], weights="distance")[0] == pytest.approx(0.0)


def test_exact_match_avoids_division_by_zero_for_weighted_regression():
    model = KNNRegressor(k=2).fit([[0], [1], [2]], [7, 10, 20])
    assert model.predict([[0]], weights="distance")[0] == pytest.approx(7.0)


def test_invalid_k_and_minkowski_parameter_are_rejected():
    with pytest.raises(ValueError):
        KNNClassifier(k=0)
    with pytest.raises(ValueError):
        pairwise_distances([[0]], [1], "minkowski", p=0.5)
