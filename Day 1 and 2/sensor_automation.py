import csv
import json
import random
import time
from datetime import datetime
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
LOG_FILE = OUTPUT_DIR / "sensor_results.csv"

FEATURES = ["temperature", "pressure", "vibration", "rpm", "current"]
TARGET = "state"

# DAY 1: Simulate readings from machine sensors.
def generate_sensor_data():
    return {
        "temperature": round(random.uniform(20, 100), 2),
        "pressure": round(random.uniform(1, 10), 2),
        "vibration": round(random.uniform(0.1, 10), 2),
        "rpm": random.randint(500, 3000),
        "current": round(random.uniform(1, 25), 2),
    }

# DAY 1: Rules are checked in priority order (smaller number = higher priority).
RULES = [
    {"name": "HIGH_TEMP", "priority": 1,
     "condition": lambda r: r["temperature"] is not None and r["temperature"] > 80,
     "action": "STOP_MACHINE"},
    {"name": "HIGH_VIBRATION", "priority": 2,
     "condition": lambda r: r["vibration"] is not None and r["vibration"] > 8.5,
     "action": "STOP_MACHINE"},
    {"name": "HIGH_PRESSURE", "priority": 3,
     "condition": lambda r: r["pressure"] is not None and r["pressure"] > 8.5,
     "action": "WARNING"},
    {"name": "HIGH_CURRENT", "priority": 4,
     "condition": lambda r: r["current"] is not None and r["current"] > 20,
     "action": "WARNING"},
    {"name": "MED_TEMP", "priority": 5,
     "condition": lambda r: r["temperature"] is not None and r["temperature"] > 65,
     "action": "REPAIR_MACHINE"},
    {"name": "LOW_PRESSURE", "priority": 6,
     "condition": lambda r: r["pressure"] is not None and r["pressure"] < 2,
     "action": "REPAIR_MACHINE"},
]

def evaluate_rules(reading, rules=RULES):
    for rule in sorted(rules, key=lambda item: item["priority"]):
        if rule["condition"](reading):
            return rule["name"], rule["action"]
    return "NO_RULE_FIRED", "CONTINUE_NORMAL"

def take_action(action):
    actions = {
        "STOP_MACHINE": "EMERGENCY! Stop machine.",
        "REPAIR_MACHINE": "Maintenance required.",
        "WARNING": "Warning! Check machine immediately.",
        "CONTINUE_NORMAL": "Machine is operating normally.",
    }
    return actions.get(action, "Unknown action.")

def log_decision(reading, rule_name, action, prediction):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reading": reading,
        "rule": rule_name,
        "action": action,
        "decision_tree_prediction": prediction,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

# DAY 2: Train and evaluate a Decision Tree using the supplied CSV.
def train_model():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\nKeep the data folder beside this script."
        )

    data = pd.read_csv(DATA_FILE)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("CSV is missing columns: " + ", ".join(missing))
    if data[required].isnull().any().any():
        raise ValueError("Dataset contains missing values. Please clean the CSV first.")
    if data[TARGET].nunique() < 2:
        raise ValueError("At least two state classes are required.")

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    report = classification_report(y_test, predictions, zero_division=0)
    importance_text = "".join(
        f"{name}: {importance:.3f}\n"
        for name, importance in sorted(
            zip(FEATURES, model.feature_importances_),
            key=lambda item: item[1], reverse=True
        )
    )
    REPORT_FILE.write_text(
        f"Dataset: {DATA_FILE.name}\nRows: {len(data)}\n"
        f"Training rows: {len(X_train)}\nTest rows: {len(X_test)}\n"
        f"Features: {', '.join(FEATURES)}\nTarget: {TARGET}\n"
        f"Accuracy: {accuracy:.3f}\n\nClassification report:\n{report}\n"
        f"Feature importances:\n{importance_text}",
        encoding="utf-8"
    )
    print("\nDAY 2: DECISION TREE TRAINING")
    print(f"Rows: {len(data)} | Training: {len(X_train)} | Testing: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(report)
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Report saved to: {REPORT_FILE}")
    return model

# DAY 1 + DAY 2:
def main():
    model = train_model()

    print("\nCOMBINED SENSOR MONITORING")
    print("=" * 60)

    for tick in range(1, 11):
        reading = generate_sensor_data()

        # DAY 2: Machine Learning output
        model_input = pd.DataFrame(
            [[reading[name] for name in FEATURES]],
            columns=FEATURES
        )
        ml_result = model.predict(model_input)[0]

        # DAY 1: Rule-based output
        rule_name, action = evaluate_rules(reading)
        rule_result = take_action(action)

        print(f"\nReading {tick}:")
        print("-" * 60)
        print("SENSOR DATA:", reading)

        print("\n1. MACHINE LEARNING OUTPUT")
        print("ML Prediction:", ml_result)

        print("\n2. RULE-BASED OUTPUT")
        print("Rule Triggered:", rule_name)
        print("Rule Action:", action)
        print("Rule Result:", rule_result)

        # ML output is selected as the final result
        final_result = ml_result

        print("\n3. FINAL RESULT")
        print("Final Machine State:", final_result)
        print("=" * 60)

        # Log both results
        log_decision(
            reading,
            rule_name,
            action,
            str(ml_result)
        )

        time.sleep(0.2)

    print("\nMonitoring completed.")
    print(f"Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
