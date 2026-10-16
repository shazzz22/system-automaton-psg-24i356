# Short written explanation

## Safety thresholds
In this synthetic demo, the arm stops immediately if torque is at least 95 Nm, temperature is at least 90 °C, or vibration is at least 12 mm/s. I set these limits high on purpose so they separate normal readings from clearly abnormal ones and show a fixed safety rule at work. They aren't general industrial limits. Real ones have to come from the robot manufacturer's specifications, a hazard analysis, and a qualified safety engineer. The stop check runs separately from the model, so it still works if training or prediction fails.

## Combining the model and the maintenance notes
A Random Forest is trained on 240 synthetic historical joint readings, each labelled NORMAL or WEAR. It gives a wear probability for each joint, and the highest one becomes the arm's score. Technician notes are scanned separately for words like grinding, backlash, bearing noise, and lubrication leak.

The decision follows a fixed order. A dangerous sensor reading or an emergency note always wins and gives EMERGENCY_STOP. Suspicious sensor readings give MONITOR so someone can check them. Signs of wear from the model, the notes, or both give SCHEDULE_MAINTENANCE. Otherwise the arm is SAFE or MONITOR, depending on its wear probability.

## Defending against spoofed sensors
The demo flags values outside a plausible range, a torque change of more than 45 Nm from the previous sample, and one cross-sensor mismatch: high torque, almost no vibration, and a cool joint. That can catch crude spoofing that produces impossible or inconsistent readings. It can't guarantee catching carefully chosen in-range values, slow drift, several sensors manipulated together, replayed data, or compromised firmware. A real system would add authenticated communication, secure timestamps, independent redundant sensors, physics-based consistency checks, and certified safety hardware. This project is for learning only and must not control real industrial equipment.
