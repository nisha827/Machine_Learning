# Machine Learning: Implementations and Experiments

A collection of machine-learning implementations and experiments covering supervised learning, classification, regression, and model evaluation.

The repository combines a refreshed, reproducible Adult Income classification baseline with historical implementations of classical machine-learning algorithms.

## Projects

### 1. Adult Income Classification

A classification experiment using the UCI Adult (Census Income) dataset.

The refreshed baseline compares logistic regression, class-weighted logistic regression, and random forest, with random oversampling evaluated as a training-only intervention. It is designed to keep preprocessing and resampling within training pipelines and evaluate models against an untouched, stratified test set.

The experiment reports:

- Accuracy
- Precision, recall, and F1
- ROC-AUC
- Confusion matrix

See [Adult Income Classification](Adult_Income_Dataset_Classification_ML_Models/README.md) for dataset requirements, methodology, setup commands, tests, and limitations.

### 2. K-Nearest Neighbours from Scratch

The repository also contains exploratory K-Nearest Neighbours (KNN) implementations for classification and regression.

Topics explored include:

- Euclidean, Manhattan, and Minkowski distance
- Neighbour selection
- Distance-weighted voting
- Feature normalisation
- Hyperparameter selection

These implementations are useful for examining the mechanics of distance-based learning. They are historical coursework code and should be reviewed and tested before being treated as a validated benchmark. In particular, preprocessing must be fitted on training data and then applied unchanged to validation and test data to avoid data leakage.

## Reproducibility

The Adult Income baseline documents its dataset schema, preprocessing, stratified split, training-only resampling, evaluation metrics, and test commands in its project README.

Some of the older coursework scripts may rely on local dataset paths or assumptions that need to be made explicit before they can be reproduced on another machine.

## Further work

- Add a consistent environment and dependency setup across projects.
- Add automated tests for the custom KNN implementations.
- Validate preprocessing and distance calculations against reference implementations.
- Compare algorithms using fixed splits or repeated cross-validation where appropriate.
- Document model performance only after experiments have been run and independently checked.
- Add error analysis and discuss limitations, including class imbalance and sensitivity to feature scaling.
