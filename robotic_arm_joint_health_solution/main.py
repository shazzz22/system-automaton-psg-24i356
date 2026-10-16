"""
Robotic arm joint health monitoring system.
For learning only. Don't use it to control real industrial equipment.

Steps:
1. Fixed emergency thresholds, checked without the ML model.
2. A Random Forest wear classifier trained on synthetic labelled history.
3. Keyword search over technician notes.
4. Spoof checks: impossible values, sudden jumps, mismatched sensors.
5. One final decision per arm.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

try:
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Run: pip install -r requirements.txt"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"
DATASET_FILE = DATA_DIR / "historical_joint_readings.csv"
MAINTENANCE_LOG = LOG_DIR / "maintenance_notes.txt"
MODEL_FILE = OUTPUT_DIR / "wear_model.joblib"
REPORT_FILE = OUTPUT_DIR / "training_report.txt"

# Example limits for this synthetic exercise.
# Real thresholds come from the robot manufacturer and a safety engineer.
TORQUE_EMERGENCY = 95.0       # Nm
TEMP_EMERGENCY = 90.0         # degrees C
VIBRATION_EMERGENCY = 12.0    # mm/s (illustrative)
TORQUE_MAX_PLAUSIBLE = 110.0
TEMP_MIN_PLAUSIBLE = -10.0
TEMP_MAX_PLAUSIBLE = 125.0
VIBRATION_MAX_PLAUSIBLE = 25.0

FEATURES = ["torque_nm", "temperature_c", "vibration_mms"]
LABELS = {0: "NORMAL", 1: "WEAR"}
MAINTENANCE_KEYWORDS = (
    "grinding", "overheat", "overheating", "unusual vibration",
    "loose joint", "backlash", "bearing noise", "lubrication leak",
    "sticking", "metal shavings", "jerking", "wear"
)
EMERGENCY_NOTE_KEYWORDS = (
    "smoke", "sparks", "burning smell", "joint seized", "uncontrolled motion"
)


def make_synthetic_dataset(path: Path = DATASET_FILE, rows: int = 240, seed: int = 42) -> None:
    """Write labelled historical readings. The values are made up, not field data."""
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["arm_id", "joint", *FEATURES, "wear_label"]
        )
        writer.writeheader()
        for i in range(rows):
            worn = 1 if rng.random() < 0.42 else 0
            if worn:
                torque = rng.gauss(76, 11)
                temp = rng.gauss(72, 9)
                vibration = rng.gauss(8.2, 2.2)
            else:
                torque = rng.gauss(48, 9)
                temp = rng.gauss(43, 8)
                vibration = rng.gauss(3.1, 1.1)
            writer.writerow({
                "arm_id": f"ARM-{(i % 6) + 1:02d}",
                "joint": rng.choice(["shoulder", "elbow", "wrist"]),
                "torque_nm": round(max(5, torque), 2),
                "temperature_c": round(max(15, temp), 2),
                "vibration_mms": round(max(0.1, vibration), 2),
                "wear_label": worn,
            })


def create_sample_maintenance_log(path: Path = MAINTENANCE_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = [
        "[2026-10-01] ARM-01 shoulder: unusual vibration noticed during fast movement.",
        "[2026-10-02] ARM-02 elbow: routine inspection completed; no issue found.",
        "[2026-10-03] ARM-03 wrist: grinding sound and possible bearing noise.",
        "[2026-10-04] ARM-04: operator cleaned the external guard; no joint issue.",
        "[2026-10-05] ARM-05 shoulder: joint temperature was higher than usual.",
        "[2026-10-06] ARM-06 elbow: loose joint and visible backlash; schedule inspection.",
        "[2026-10-07] ARM-01: production count was 1200 units; packaging label updated.",
        "[2026-10-08] ARM-02 wrist: lubrication leak reported near the joint.",
        "[2026-10-09] ARM-03: smoke and burning smell reported near the arm.",
        "[2026-10-10] ARM-04: weekly meeting moved to 10:30; no equipment issue.",
    ]
    path.write_text("\n".join(notes) + "\n", encoding="utf-8")


def train_model(dataset_path: Path = DATASET_FILE):
    df = pd.read_csv(dataset_path)
    required = set(FEATURES + ["wear_label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("Dataset contains missing feature or label values.")
    if df["wear_label"].nunique() < 2:
        raise ValueError("Training data must contain both wear labels.")

    X = df[FEATURES]
    y = df["wear_label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=120, max_depth=7, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    REPORT_FILE.write_text(
        f"Synthetic dataset rows: {len(df)}\n"
        f"Test accuracy: {accuracy_score(y_test, predictions):.3f}\n\n"
        f"{classification_report(y_test, predictions, target_names=['NORMAL', 'WEAR'])}",
        encoding="utf-8",
    )
    return model


def scan_maintenance_notes(arm_id: str, log_path: Path = MAINTENANCE_LOG) -> dict[str, Any]:
    if not log_path.exists():
        return {"risk": False, "emergency_note": False, "matched": [], "notes": []}
    matching_lines = [
        line.strip() for line in log_path.read_text(encoding="utf-8").splitlines()
        if arm_id.lower() in line.lower()
    ]
    text = " ".join(matching_lines).lower()
    matched = [word for word in MAINTENANCE_KEYWORDS if word in text]
    emergency = [word for word in EMERGENCY_NOTE_KEYWORDS if word in text]
    return {
        "risk": bool(matched),
        "emergency_note": bool(emergency),
        "matched": matched,
        "notes": matching_lines,
    }


def detect_spoofing(reading: dict[str, Any], previous_reading: dict[str, Any] | None = None) -> list[str]:
    """Simple anomaly checks. They catch obvious inconsistencies, not every attack."""
    flags = []
    torque = float(reading["torque_nm"])
    temp = float(reading["temperature_c"])
    vibration = float(reading["vibration_mms"])

    if torque < 0 or torque > TORQUE_MAX_PLAUSIBLE:
        flags.append("torque outside plausible sensor range")
    if temp < TEMP_MIN_PLAUSIBLE or temp > TEMP_MAX_PLAUSIBLE:
        flags.append("temperature outside plausible sensor range")
    if vibration < 0 or vibration > VIBRATION_MAX_PLAUSIBLE:
        flags.append("vibration outside plausible sensor range")

    # Torque shouldn't swing this far between two consecutive samples.
    if previous_reading is not None:
        old_torque = float(previous_reading["torque_nm"])
        if abs(torque - old_torque) > 45:
            flags.append("implausibly large torque jump from previous reading")

    # High torque with almost no vibration on a cool joint usually means the
    # vibration sensor is disconnected or stuck, or the readings don't belong together.
    if torque > 85 and vibration < 0.4 and temp < 45:
        flags.append("cross-sensor readings are inconsistent")
    return flags


def immediate_safety_stop(reading: dict[str, Any]) -> list[str]:
    """Fixed threshold rules. This never touches the ML model."""
    reasons = []
    if float(reading["torque_nm"]) >= TORQUE_EMERGENCY:
        reasons.append(f"torque >= {TORQUE_EMERGENCY} Nm")
    if float(reading["temperature_c"]) >= TEMP_EMERGENCY:
        reasons.append(f"temperature >= {TEMP_EMERGENCY} C")
    if float(reading["vibration_mms"]) >= VIBRATION_EMERGENCY:
        reasons.append(f"vibration >= {VIBRATION_EMERGENCY} mm/s")
    return reasons


def assess_arm(
    arm_id: str,
    readings: list[dict[str, Any]],
    model,
    log_path: Path = MAINTENANCE_LOG,
    previous_by_joint: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    previous_by_joint = previous_by_joint or {}
    emergency_reasons = []
    spoof_flags = []
    wear_scores = []
    for reading in readings:
        joint = reading.get("joint", "unknown")
        emergency_reasons.extend(
            f"{joint}: {reason}" for reason in immediate_safety_stop(reading)
        )
        spoof_flags.extend(
            f"{joint}: {flag}"
            for flag in detect_spoofing(reading, previous_by_joint.get(joint))
        )
        features = [[float(reading[name]) for name in FEATURES]]
        wear_scores.append(float(model.predict_proba(features)[0][1]))

    notes = scan_maintenance_notes(arm_id, log_path)
    max_wear_probability = max(wear_scores, default=0.0)

    # Order: safety stop, then spoof warning, then maintenance evidence, then the model alone.
    if emergency_reasons or notes["emergency_note"]:
        decision = "EMERGENCY_STOP"
        reason = "Dangerous sensor threshold or emergency technician note."
    elif spoof_flags:
        decision = "MONITOR"
        reason = "Potential spoofing or implausible sensor readings need verification."
    elif notes["risk"] and max_wear_probability >= 0.45:
        decision = "SCHEDULE_MAINTENANCE"
        reason = "Wear prediction and maintenance-note evidence agree."
    elif notes["risk"] or max_wear_probability >= 0.72:
        decision = "SCHEDULE_MAINTENANCE"
        reason = "Strong wear prediction or maintenance note indicates service is needed."
    elif max_wear_probability >= 0.40:
        decision = "MONITOR"
        reason = "Wear probability is elevated; collect more readings."
    else:
        decision = "SAFE"
        reason = "No emergency, spoof flag, note risk, or elevated wear probability."

    return {
        "arm_id": arm_id,
        "decision": decision,
        "reason": reason,
        "max_wear_probability": round(max_wear_probability, 3),
        "maintenance_note_matches": notes["matched"],
        "maintenance_notes": notes["notes"],
        "spoof_flags": spoof_flags,
        "emergency_reasons": emergency_reasons,
    }


def example_readings() -> dict[str, list[dict[str, Any]]]:
    def r(joint, torque, temp, vibration):
        return {"joint": joint, "torque_nm": torque, "temperature_c": temp, "vibration_mms": vibration}
    # One arm per outcome. ARM-06 also has a backlash note in the maintenance log.
    return {
        "ARM-10": [r("shoulder", 44, 40, 2.5), r("elbow", 47, 42, 3.0), r("wrist", 43, 39, 2.4)],
        # The shoulder sensors disagree on purpose, so this gives MONITOR whatever the model says.
        "ARM-11": [r("shoulder", 90, 40, 0.1), r("elbow", 64, 57, 5.5), r("wrist", 66, 58, 5.8)],
        "ARM-06": [r("shoulder", 78, 74, 8.5), r("elbow", 72, 68, 7.2), r("wrist", 69, 65, 6.9)],
        "ARM-12": [r("shoulder", 99, 65, 8.0), r("elbow", 45, 42, 3.0), r("wrist", 43, 40, 2.5)],
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_FILE.exists():
        make_synthetic_dataset()
    if not MAINTENANCE_LOG.exists():
        create_sample_maintenance_log()

    model = train_model()
    print(f"Model trained and saved to: {MODEL_FILE}")
    print(f"Training report saved to: {REPORT_FILE}")
    print("\nExample decisions (synthetic readings):")
    for arm_id, readings in example_readings().items():
        result = assess_arm(arm_id, readings, model)
        print(json.dumps(result, indent=2))
    print("\nNote: thresholds and sample data are educational only, not validated for real machinery.")


if __name__ == "__main__":
    main()
