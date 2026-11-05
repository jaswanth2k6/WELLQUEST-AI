
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

DATA_PATH = Path("data/StressLevelDataset.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nWELLQUEST AI - MODEL TRAINING")
print("=" * 45)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. DATA VALIDATION
# --------------------------------------------------

if "stress_level" not in df.columns:
    raise ValueError("Target column stress_level not found!")

if df.isnull().sum().sum() > 0:
    raise ValueError("Missing values detected. Clean data first.")

if df.duplicated().sum() > 0:
    raise ValueError("Duplicate records detected.")

# --------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=["stress_level"])
y = df["stress_level"]

feature_names = X.columns.tolist()
class_labels = sorted(y.unique().tolist())

print("\nNumber of input features:", len(feature_names))
print("Stress classes:", class_labels)

# --------------------------------------------------
# 5. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 6. DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
}

results = {}

# --------------------------------------------------
# 7. TRAIN AND EVALUATE
# --------------------------------------------------

for name, model in models.items():

    print("\n" + "=" * 45)
    print("TRAINING:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results[name] = {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4)
    }

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1-score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Save confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=class_labels)

    pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in class_labels],
        columns=[f"Predicted_{c}" for c in class_labels]
    ).to_csv(REPORT_DIR / f"{name.replace(' ', '_')}_confusion_matrix.csv")

    # Save trained model
    joblib.dump(
        model,
        MODEL_DIR / f"{name.replace(' ', '_').lower()}.joblib"
    )

    # Save feature importance for Random Forest
    if name == "Random Forest":
        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values(
            by="importance",
            ascending=False
        )

        importance_df.to_csv(
            REPORT_DIR / "feature_importance.csv",
            index=False
        )

# --------------------------------------------------
# 8. SAVE MODEL COMPARISON
# --------------------------------------------------

with open(REPORT_DIR / "model_comparison.json", "w") as file:
    json.dump(results, file, indent=4)

print("\n" + "=" * 45)
print("MODEL COMPARISON")
print("=" * 45)

for name, scores in results.items():
    print("\n", name)
    for metric, value in scores.items():
        print(metric, ":", value)

print("\nAll models and reports saved successfully.")
print("WELLQUEST AI training completed.")
