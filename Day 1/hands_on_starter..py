"""
DAY 1 · HANDS-ON — Build Your Rule Engine
23IF02 System Automation

Work through the TODOs in order. Each one matches a step on the
"Hands-on · Build Your Rule Engine" slide. Run this file after each
step to see your progress:

    python hands_on_starter.py

Need the answer? See hands_on_solution.py — but try it yourself first.
"""

import random
import time
import json
from datetime import datetime

LOG_FILE = "day1_log.txt"


# ---------------------------------------------------------------------------
# STEP 1 — Generate simulated sensor data
# ---------------------------------------------------------------------------
def generate_sensor_data():
    """Return a dict with at least 'temperature', 'pressure', 'vibration'."""
    # TODO: fill in realistic random values, e.g. temperature 50-95 (deg C)
    return {
        "temperature": random.uniform(50,95),   
        "pressure": random.uniform(1.0,10.0),     
        "vibration": random.uniform(0.0,5.0),  
        "current": random.uniform(0.5,25.0)   
    }
    # STRETCH GOAL (after the 8 steps below all work): real sensors aren't
    # perfectly clean. Try making one field randomly return None (a
    # dropout) or spike far outside its normal range every so often, then
    # update evaluate_rules() so a dropout doesn't crash your engine.


# ---------------------------------------------------------------------------
# STEP 2 — Write your first automation rules
# STEP 7 — Add a rule of your own (add another dict to this list)
# STEP 8 — Add one priority rule (lower "priority" number checked first)
# ---------------------------------------------------------------------------
RULES = [
    {"name": "HIGH_TEMP","priority": 1,"condition": lambda r: r["temperature"] is not None and r["temperature"] > 80, "action": "STOP_MACHINE"},
    {"name": "MED_TEMP","priority": 5,"condition": lambda r: r["temperature"] is not None and r["temperature"] > 50,"action": "REPAIR_MACHINE"},
    {"name": "LOW_TEMP","priority": 6,"condition": lambda r: r["temperature"] is not None and r["temperature"] > 15,"action": "CONTINUE"},
    {"name": "HIGH_VIBRATION","priority": 2,"condition": lambda r: r["vibration"] is not None and r["vibration"] > 8.5, "action": "STOP_MACHINE"},
    {"name": "HIGH_PRESSURE","priority": 3,"condition": lambda r: r["pressure"] is not None and r["pressure"] > 3.5,"action": "STOP"},
    {"name": "LOW_PRESSURE","priority": 7,"condition": lambda r: r["pressure"] is not None and r["pressure"] < 1.0,"action": "REPAIR_MACHINE"},
    {"name": "HIGH_CURRENT","priority": 4,"condition": lambda r: r["current"] is not None and r["current"] >20.0,"action": "WARNING"}
]


# ---------------------------------------------------------------------------
# STEP 3 — Detect machine state from the rules
# ---------------------------------------------------------------------------
def evaluate_rules(reading, rules):
    rules=sorted(rules,key=lambda x:x['priority'])
    for i in rules:
        if i["condition"](reading):
            return i['name'],i['action']

    """Return (rule_name, action) for the first matching rule, checked in
    priority order (lowest number first). If nothing matches, return
    ("NO_RULE_FIRED", "CONTINUE_NORMAL")."""
    # TODO: sort `rules` by priority, then check each rule's condition(reading)
    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# ---------------------------------------------------------------------------
# STEP 4 — Trigger an action when a rule fires
# ---------------------------------------------------------------------------
def take_action(action):
    """Return a human-readable string describing what happens for this action."""
    actions = {"STOP_MACHINE": "EMERGENCY! Stop machine.",
        "REPAIR_MACHINE": "Maintenance required.",
        "CONTINUE": "Continue normal Operation",
        "STOP": "HIGH PRESSURE ALERT!, Reduce the pressure immediately.",
        "WARNING": "Current exceeding, action required!"}

    return actions.get(action, f"(no handler yet for action: {action})")


# ---------------------------------------------------------------------------
# STEP 5 — Log every decision made
# ---------------------------------------------------------------------------
def log_decision(reading, rule_name, action):
    """Append one line to LOG_FILE with the reading, rule fired, and action."""
    log_entry = {"timestamp": datetime.now().isoformat(timespec="seconds"),"reading": reading,"rule": rule_name,"action": action}

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(log_entry) + "\n")


# ---------------------------------------------------------------------------
# STEP 6 — Add a second sensor to the system
# TODO: add e.g. "current" to generate_sensor_data(), then use it in a rule
# ---------------------------------------------------------------------------


def main():
    for tick in range(1, 11):
        reading = generate_sensor_data()
        rule_name, action = evaluate_rules(reading, RULES)
        outcome = take_action(action)

        print(f"[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}\n")

        log_decision(reading, rule_name, action)
        time.sleep(0.2)


if __name__ == "__main__":
    main()

