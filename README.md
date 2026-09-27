# Heart Failure Clinical Records — Machine Learning Evaluation

This repository contains a reproducible machine-learning project using the UCI Heart Failure Clinical Records dataset (Dataset 519).

## Objective
Implement supervised ML models on the same dataset, evaluate them, and compare the results with the published benchmark from Chicco & Jurman (2020).

## Dataset
- UCI Machine Learning Repository, Dataset 519
- 299 patient records
- 12 predictors
- Target: `DEATH_EVENT`
- UCI reports no missing values

Dataset: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

## Models
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM (RBF)

## Evaluation
A reproducible 80/20 stratified hold-out split (`random_state=42`) is used, with additional 5-fold stratified cross-validation on the training set.

Metrics:
Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, and ROC curves.

## Published benchmark
Chicco & Jurman (2020) reported, for all clinical features:
- Random Forest: Accuracy 0.740, ROC-AUC 0.800, F1 0.547, MCC 0.384
- Decision Tree: Accuracy 0.737, ROC-AUC 0.681, F1 0.554, MCC 0.376
- Gradient Boosting: Accuracy 0.738, ROC-AUC 0.754, F1 0.527, MCC 0.367
- SVM radial: Accuracy 0.690, ROC-AUC 0.749, F1 0.182, MCC 0.159

The paper used repeated randomized experiments, so the local implementation is a reproducible benchmark comparison rather than an identical experimental replication.

## Run
```bash
pip install -r requirements.txt
python src/train.py
```

The script downloads the dataset from UCI when network access is available and writes generated metrics/plots to `results/`.

## References
1. UCI Machine Learning Repository. Heart Failure Clinical Records, Dataset 519. DOI: 10.24432/C5Z89R.
2. Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making, 20, 16. DOI: 10.1186/s12911-020-1023-5.
