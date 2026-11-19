# K-Nearest Neighbours: Maintained Implementation

The original 2018 scripts are retained as historical coursework examples. Use
`knn.py` for new experiments. It provides:

- KNN classification and regression
- Euclidean, Manhattan and Minkowski distances
- Train-fitted min-max scaling (fit on training data, then transform test data)
- Deterministic tie handling
- Exact-match handling for distance-weighted regression
- Input validation and import-safe modules

## Run tests

From the repository root:

```bash
python -m pip install numpy pytest
python -m pytest k_Nearest_Neighbors_ML_Model/tests
```

## Example

```python
from k_Nearest_Neighbors_ML_Model.knn import KNNClassifier, MinMaxScaler

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNNClassifier(k=5, metric="minkowski", p=3).fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
```

Fit preprocessing only on training data. Choose k and distance metrics using
training/validation data; keep the test set for final evaluation. This module
does not claim performance on a particular dataset.
