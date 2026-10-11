"""Train and evaluate a decision tree on the Day 2 sensor dataset."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "day2_sample.csv"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_FILE = OUTPUT_DIR / "decision_tree_model.joblib"
REPORT_FILE = OUTPUT_DIR / "training_report.txt"

FEATURES = ["temperature", "pressure", "vibration", "rpm", "current"]
TARGET = "state"


def main():
    data = pd.read_csv(DATA_FILE)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    if data[required_columns].isnull().any().any():
        raise ValueError("The sensor data contains missing values; clean them before training.")
    if data[TARGET].nunique() < 2:
        raise ValueError("At least two state classes are required for training.")

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    REPORT_FILE.write_text(
        f"Dataset: {DATA_FILE.name}\n"
        f"Rows: {len(data)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Test rows: {len(X_test)}\n"
        f"Features: {', '.join(FEATURES)}\n"
        f"Target: {TARGET}\n"
        f"Accuracy: {accuracy:.3f}\n\n"
        f"Classification report:\n{report}\n"
        f"Feature importances:\n"
        + "".join(
            f"{name}: {importance:.3f}\n"
            for name, importance in sorted(
                zip(FEATURES, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        encoding="utf-8",
    )

    print(f"Trained decision tree on {len(X_train)} rows; test accuracy: {accuracy:.3f}")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Evaluation report saved to: {REPORT_FILE}")
    print(report)


if __name__ == "__main__":
    main()