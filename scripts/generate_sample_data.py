#!/usr/bin/env python3
"""Generate sample_sensor_data.csv with the real 8-sensor AayuSense schema.

Columns: timestamp, pH, TDS, orp_mV, turbidity, Reduction_value,
         Ionic_value, Salt_content, temp_c, quality_label

Classes (4): Genuine, Adulterated_Starch, Adulterated_Synthetic_Dye, Adulterated_Heavy_Metal
125 samples per class = 500 total.

Sensor ranges match the AayuSense hardware calibration:
  pH:              0–14    (Genuine herbs: 5.5–7.5)
  TDS:             0–1000  (Genuine: 150–400 ppm)
  orp_mV:          -500–500 (Genuine: 180–380 mV)
  turbidity:       0–1.0   (Genuine: 0.05–0.25)
  Reduction_value: 0–1.0   (Genuine: 0.08–0.25)
  Ionic_value:     0–1.0   (Genuine: 0.05–0.18)
  Salt_content:    0–1.0   (Genuine: 0.10–0.30)
  temp_c:          20–35   (all classes: 23–28 °C)
"""

import csv
import random
import math
from datetime import datetime, timedelta

random.seed(42)

CLASSES = {
    "Genuine": {
        "pH":              (5.5, 7.5, 0.3),
        "TDS":             (150, 400, 25),
        "orp_mV":          (180, 380, 20),
        "turbidity":       (0.05, 0.25, 0.02),
        "Reduction_value": (0.08, 0.25, 0.02),
        "Ionic_value":     (0.05, 0.18, 0.015),
        "Salt_content":    (0.10, 0.30, 0.02),
        "temp_c":          (23.0, 27.0, 0.5),
    },
    "Adulterated_Starch": {
        # Starch adulterant: elevates TDS, turbidity; pH drops slightly
        "pH":              (4.8, 6.2, 0.4),
        "TDS":             (400, 700, 40),
        "orp_mV":          (80, 200, 25),
        "turbidity":       (0.35, 0.70, 0.05),
        "Reduction_value": (0.05, 0.15, 0.02),
        "Ionic_value":     (0.08, 0.22, 0.02),
        "Salt_content":    (0.20, 0.45, 0.03),
        "temp_c":          (23.5, 27.5, 0.5),
    },
    "Adulterated_Synthetic_Dye": {
        # Synthetic dye: high ORP, low pH, distinct ionic signature
        "pH":              (3.5, 5.0, 0.5),
        "TDS":             (250, 550, 35),
        "orp_mV":          (350, 490, 30),
        "turbidity":       (0.08, 0.30, 0.03),
        "Reduction_value": (0.15, 0.40, 0.03),
        "Ionic_value":     (0.18, 0.40, 0.03),
        "Salt_content":    (0.08, 0.22, 0.02),
        "temp_c":          (23.0, 27.5, 0.5),
    },
    "Adulterated_Heavy_Metal": {
        # Heavy metal: very high TDS, high salt, low ORP
        "pH":              (6.5, 8.5, 0.4),
        "TDS":             (600, 950, 50),
        "orp_mV":          (-100, 120, 30),
        "turbidity":       (0.20, 0.55, 0.04),
        "Reduction_value": (0.30, 0.65, 0.05),
        "Ionic_value":     (0.25, 0.55, 0.04),
        "Salt_content":    (0.40, 0.80, 0.05),
        "temp_c":          (23.0, 28.0, 0.5),
    },
}

N_PER_CLASS = 125
START_TIME = datetime(2025, 1, 15, 0, 0, 0)


def gauss_clamp(lo, hi, sigma):
    mid = (lo + hi) / 2
    val = random.gauss(mid, sigma)
    return max(lo, min(hi, val))


rows = []
for label, params in CLASSES.items():
    for i in range(N_PER_CLASS):
        row = {}
        row["pH"] = round(gauss_clamp(*params["pH"]), 3)
        row["TDS"] = round(gauss_clamp(*params["TDS"]), 2)
        row["orp_mV"] = round(gauss_clamp(*params["orp_mV"]), 2)
        row["turbidity"] = round(gauss_clamp(*params["turbidity"]), 4)
        row["Reduction_value"] = round(gauss_clamp(*params["Reduction_value"]), 4)
        row["Ionic_value"] = round(gauss_clamp(*params["Ionic_value"]), 4)
        row["Salt_content"] = round(gauss_clamp(*params["Salt_content"]), 4)
        row["temp_c"] = round(gauss_clamp(*params["temp_c"]), 2)
        row["quality_label"] = label
        rows.append(row)

# Shuffle and assign sequential timestamps
random.shuffle(rows)
for idx, row in enumerate(rows):
    ts = START_TIME + timedelta(minutes=idx * 10)
    row["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")

# Sort by timestamp
rows.sort(key=lambda r: r["timestamp"])

FIELDNAMES = [
    "timestamp", "pH", "TDS", "orp_mV", "turbidity",
    "Reduction_value", "Ionic_value", "Salt_content", "temp_c", "quality_label"
]

OUTPUT_PATH = r"c:\Users\UMESH PANDEY\OneDrive\Documents\Adobe\github-portfolio\AayuSense-AI-ETongue\data\raw\sample_sensor_data.csv"

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows to {OUTPUT_PATH}")

# Verify class distribution
from collections import Counter
dist = Counter(r["quality_label"] for r in rows)
for cls, cnt in dist.items():
    print(f"  {cls}: {cnt}")
