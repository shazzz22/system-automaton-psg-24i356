COMBINED DAY 1 + DAY 2 SENSOR AUTOMATION PROJECT

Files:
- combined_sensor_automation.py : rule engine + Decision Tree training/testing + live predictions
- data/day2_sample.csv          : Day 2 dataset
- output/                       : created automatically when the script runs

Setup:
1. Install Python.
2. Open Command Prompt in this folder.
3. Run: pip install pandas scikit-learn joblib
4. Run: python combined_sensor_automation.py

The program trains the Decision Tree from the CSV, reports test accuracy,
then generates 10 simulated sensor readings. Each reading is checked by
the Day 1 priority rule engine and the Day 2 trained model. Results are
saved in output/day1_log.txt; the model and report are also saved there.
