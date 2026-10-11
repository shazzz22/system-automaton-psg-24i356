"""Generate labeled synthetic sensor readings for the Day 2 classifier."""

import csv
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "generated_sensor_readings.csv"
READINGS_PER_STATE = 50
RANDOM_SEED = 42

FIELDS = ["temperature", "pressure", "vibration", "rpm", "current", "state"]


def generate_reading(state, rng):
    """Create one synthetic reading with values typical of the given state."""
    # Keep the non-target readings in plausible ranges while making the main
    # sensor associated with each state clearly different for demonstration.
    temperature = rng.uniform(45, 75)
    pressure = rng.uniform(4.0, 7.0)
    vibration = rng.uniform(0.5, 3.5)
    rpm = rng.randint(1200, 2500)
    current = rng.uniform(5.0, 15.0)

    if state == "OVERHEATING":
        temperature = rng.uniform(80, 100)
    elif state == "LOW_PRESSURE":
        pressure = rng.uniform(1.0, 3.0)
    elif state == "HIGH_VIBRATION":
        vibration = rng.uniform(6.0, 10.0)
    elif state == "MOTOR_OVERLOAD":
        current = rng.uniform(17.0, 30.0)

    return {
        "temperature": round(temperature, 2),
        "pressure": round(pressure, 2),
        "vibration": round(vibration, 2),
        "rpm": rpm,
        "current": round(current, 2),
        "state": state,
    }


def main():
    rng = random.Random(RANDOM_SEED)
    states = [
        "NORMAL",
        "OVERHEATING",
        "LOW_PRESSURE",
        "HIGH_VIBRATION",
        "MOTOR_OVERLOAD",
    ]
    readings = [
        generate_reading(state, rng)
        for state in states
        for _ in range(READINGS_PER_STATE)
    ]
    rng.shuffle(readings)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(readings)

    print(f"Generated {len(readings)} sensor readings.")
    print(f"CSV saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
