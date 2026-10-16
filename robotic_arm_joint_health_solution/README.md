# Robotic arm joint health monitoring system

A Python solution to the optional bonus challenge, written as a learning exercise. Don't connect it to a real robot, and don't use its thresholds for real safety decisions. Industrial safety limits have to come from the robot manufacturer and be validated by a qualified safety engineer.

## Project files

- `main.py` creates the data and log files if they're missing, trains the model, checks sensor readings, scans the maintenance notes, and prints example decisions.
- `data/historical_joint_readings.csv` holds 240 synthetic historical readings, each with a wear label.
- `logs/maintenance_notes.txt` has 10 sample technician notes. Some are about joint problems and some aren't.
- `requirements.txt` lists the Python libraries you need.
- `output/` is where the program saves the model and training report.

## Running it on Windows

1. Install Python 3.10 or later.
2. Extract the ZIP file.
3. Open Command Prompt in the extracted `robotic_arm_joint_health_solution` folder.
4. Install the dependencies:

   ```bat
   python -m pip install -r requirements.txt
   ```

5. Run the program:

   ```bat
   python main.py
   ```

It prints one decision for each sample arm and saves `output/wear_model.joblib` and `output/training_report.txt`.

## How the pipeline works

1. Immediate stop. Fixed thresholds are checked before anything else, and they don't depend on the ML model. Any breach gives `EMERGENCY_STOP`.
2. Wear prediction. A Random Forest classifier is trained on the synthetic historical readings, labelled `0 = NORMAL` and `1 = WEAR`.
3. Maintenance notes. Notes for the arm being checked are searched for keywords like `grinding`, `backlash`, `bearing noise`, and `lubrication leak`. Emergency terms such as `smoke` and `burning smell` give `EMERGENCY_STOP`.
4. Spoof checks. The program looks for impossible or out-of-range values, a sudden torque jump compared with the previous sample, and one simple cross-sensor mismatch: high torque with almost no vibration on a cool joint. A spoof flag gives `MONITOR` unless an emergency stop applies.
5. Decision order: `EMERGENCY_STOP`, then `MONITOR` for suspicious readings, then `SCHEDULE_MAINTENANCE`, then `MONITOR` for a raised wear probability, then `SAFE`.

## Why these thresholds

The limits in `main.py` (95 Nm torque, 90 °C temperature, 12 mm/s vibration) are example values picked to show how the fixed safety rules take priority. They aren't general engineering limits. A real installation needs the manufacturer's validated specifications and operating envelope, a risk assessment, and safety-rated hardware and interlocks. Software should never be the only way to trigger an emergency stop.

## How the model and the notes combine

The model gives a wear probability for each joint, and the arm's score is the highest of those. The maintenance notes are a separate source of evidence. Either a note that points to wear or a high wear probability can lead to scheduled maintenance. An emergency note or a dangerous sensor reading leads to `EMERGENCY_STOP`. Suspicious sensor data leads to `MONITOR` so an operator can check the sensors.

## What the spoof checks catch and miss

The checks flag values outside a plausible range, large sudden torque jumps, and one obvious cross-sensor mismatch, so they can pick up crude spoofing that injects an extreme or inconsistent value. They won't reliably catch a carefully chosen in-range value that drifts slowly, several sensors spoofed together, replayed data, or compromised sensor firmware. Defending against those takes authenticated sensor communication, secure timestamps or counters, independent redundant sensors, physical plausibility models, and monitoring on a trusted controller.

## The four example outcomes

The program prints an example of each label:

- `SAFE`: normal readings and no relevant maintenance note.
- `MONITOR`: a raised wear probability or suspicious sensor data.
- `SCHEDULE_MAINTENANCE`: a maintenance note or a strong wear prediction.
- `EMERGENCY_STOP`: a threshold breach or an emergency note.

The model's probabilities depend on the generated training data and on which library versions you have installed, so check the printed results when you run it. The example readings are set up to reach each path, and you can use the output to talk through any differences the model produces.

## Limitations

This is a learning exercise on synthetic data, not a validated predictive maintenance model. It doesn't model the full dynamics of a six-axis robot, and it can't replace certified safety controls.
