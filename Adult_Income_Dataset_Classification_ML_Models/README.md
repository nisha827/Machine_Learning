# Adult Income Classification

A reproducible classification experiment using the UCI Adult (Census Income) dataset. The original project compared classifiers and resampling methods. This refreshed baseline focuses first on sound evaluation: preprocessing and any resampling are fitted on the training split only, numeric and categorical predictors are both retained, and the held-out test set is never resampled.

## Experimental question

How do a standard logistic-regression baseline, class-weighted logistic regression, random forest, and random oversampling affect prediction quality on the minority `>50K` income class?

## Data

Place the headerless UCI Adult CSV at `Adult_Income_Dataset_Classification_ML_Models/adult.csv`, or pass a path with `--data`. The loader expects the standard 15-column Adult schema, strips whitespace, treats `?` as missing, and accepts both period-suffixed and unsuffixed income labels. It does not download data. Check the dataset's source and usage terms before redistributing it.

## Methodology

1. Make one stratified train/test split using a fixed random seed.
2. Fit median imputation and standard scaling for numeric columns on training data only.
3. Fit most-frequent imputation and one-hot encoding for categorical columns on training data only.
4. Compare logistic regression, class-weighted logistic regression, and random forest, each with and without random oversampling where configured. Oversampling is inside an imbalanced-learn pipeline, so it runs only during `fit` and never on test data.
5. Report accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix.

Random oversampling is used as the initial sampler because it does not synthesize fractional one-hot category combinations. SMOTE-family methods and feature selection are intentionally deferred until their handling of mixed numeric/categorical data and cross-validation are designed and tested. No performance result is claimed until the experiment is run on the dataset.

## Setup and run

From this directory:

```bash
python -m pip install -r requirements.txt
python -m src.train --seed 123 --test-size 0.25 --output results/baseline_metrics.csv
```

Or use the legacy entry point:

```bash
python my_code.py --seed 123
```

Run tests:

```bash
python -m pip install pytest
python -m pytest tests
```

Importing `my_code.py` does not launch training. Set `--data` to use a dataset stored elsewhere.

## Scope and limitations

This is a baseline comparison, not a tuned benchmark. Hyperparameter search, threshold selection, calibration, feature selection, repeated/nested cross-validation, and a broader SMOTE/ADASYN study are follow-up work. The test set should remain untouched until model choices are finalized. Dataset provenance and licensing should be documented alongside any redistributed copy.
