# Machine Learning on Heart Failure Clinical Records

## 1. Introduction
This project applies supervised machine learning to the UCI Heart Failure Clinical Records dataset. The target variable is `DEATH_EVENT`.

## 2. Dataset
The UCI dataset contains 299 patient records and 12 predictor variables. UCI reports no missing values.

## 3. Methodology
1. Load the UCI dataset.
2. Separate predictors and target.
3. Perform an 80/20 stratified train/test split with random_state=42.
4. Train Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and RBF-SVM.
5. Standardize inputs inside pipelines for Logistic Regression and SVM.
6. Evaluate on the held-out test set using accuracy, precision, recall, F1 and ROC-AUC.
7. Perform 5-fold stratified cross-validation on the training set.
8. Save metrics, confusion-matrix counts, classification reports and ROC curves.

## 4. Published comparison
Chicco & Jurman (2020) evaluated multiple classifiers using repeated randomized experiments.

| Model | Accuracy | ROC-AUC | F1 | MCC |
|---|---:|---:|---:|---:|
| Random Forest | 0.740 | 0.800 | 0.547 | 0.384 |
| Decision Tree | 0.737 | 0.681 | 0.554 | 0.376 |
| Gradient Boosting | 0.738 | 0.754 | 0.527 | 0.367 |
| SVM radial | 0.690 | 0.749 | 0.182 | 0.159 |

These numbers are the published benchmark. The local experiment uses a different split/CV protocol, so direct metric equality is not expected.

## 5. Results
Run:
```bash
python src/train.py
```
The generated `results/model_results.csv` contains the actual local results. `results/roc_curves.png` contains the ROC comparison.

## 6. Discussion
The purpose of the experiment is to demonstrate end-to-end ML implementation and reproducible evaluation. Accuracy should be interpreted together with recall, F1 and ROC-AUC because the outcome classes are not perfectly balanced.

## 7. Conclusion
The repository provides a complete, reproducible academic implementation and a sourced comparison with the 2020 benchmark.

This project is for academic experimentation and is not clinical validation or a medical decision-support system.

## References
- UCI Machine Learning Repository. Heart Failure Clinical Records, Dataset 519. DOI: 10.24432/C5Z89R.
- Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making, 20, 16. DOI: 10.1186/s12911-020-1023-5.
