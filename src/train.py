from pathlib import Path
import io, zipfile, urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/static/public/519/heart%2Bfailure%2Bclinical%2Brecords.zip"
CSV_NAME = "heart_failure_clinical_records_dataset.csv"
LOCAL_CSV = DATA_DIR / CSV_NAME

def load_data():
    if LOCAL_CSV.exists():
        return pd.read_csv(LOCAL_CSV)
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(CSV_NAME) as f:
            df = pd.read_csv(f)
    df.to_csv(LOCAL_CSV, index=False)
    return df

def main():
    df = load_data()
    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    rows = []
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, pred)
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({
            "model": name,
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "test_roc_auc": auc,
            "cv_accuracy_mean": cv_res["test_accuracy"].mean(),
            "cv_f1_mean": cv_res["test_f1"].mean(),
            "cv_roc_auc_mean": cv_res["test_roc_auc"].mean(),
            "tn": cm[0,0], "fp": cm[0,1], "fn": cm[1,0], "tp": cm[1,1],
        })
        report_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        (RESULTS_DIR / f"{report_name}_classification_report.txt").write_text(
            classification_report(y_test, pred, zero_division=0)
        )

    plt.plot([0,1], [0,1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Heart Failure Clinical Records")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=200)
    plt.close()

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()
