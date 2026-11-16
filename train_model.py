from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "SVM": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(
        kernel="rbf", C=10, gamma="scale", probability=True, random_state=42
    ))]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    rows.append({"Model": name, "Accuracy (%)": round(accuracy, 2)})
    filename = "svm_digit_model.joblib" if name == "SVM" else "random_forest_digit_model.joblib"
    joblib.dump(model, ROOT / "models" / filename)
    print(f"{name} test accuracy: {accuracy:.2f}%")

pd.DataFrame(rows).to_csv(ROOT / "reports" / "model_metrics.csv", index=False)
print("\nTraining completed. Models saved in the models folder.")
