import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pathlib

rng = np.random.default_rng(42)

start_time = datetime(2025, 1, 15, 0, 0, 0)
interval = timedelta(minutes=10)

records = []
current_time = start_time

# Genuine: 125 rows - pH 6.4-7.2, conductivity 1.0-1.4, turbidity 35-55, ORP 155-205
for _ in range(125):
    records.append({
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'ph': round(rng.uniform(6.4, 7.2), 4),
        'conductivity': round(rng.uniform(1.0, 1.4), 4),
        'turbidity': round(rng.uniform(35, 55), 4),
        'orp': round(rng.uniform(155, 205), 4),
        'quality_label': 'Genuine',
    })
    current_time += interval

# Adulterant_A: 125 rows
for _ in range(125):
    records.append({
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'ph': round(rng.uniform(4.8, 5.6), 4),
        'conductivity': round(rng.uniform(2.5, 3.1), 4),
        'turbidity': round(rng.uniform(100, 140), 4),
        'orp': round(rng.uniform(65, 125), 4),
        'quality_label': 'Adulterant_A',
    })
    current_time += interval

# Adulterant_B: 125 rows
for _ in range(125):
    records.append({
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'ph': round(rng.uniform(7.4, 8.4), 4),
        'conductivity': round(rng.uniform(0.3, 0.5), 4),
        'turbidity': round(rng.uniform(15, 35), 4),
        'orp': round(rng.uniform(270, 350), 4),
        'quality_label': 'Adulterant_B',
    })
    current_time += interval

# Adulterant_C: 125 rows
for _ in range(125):
    records.append({
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'ph': round(rng.uniform(5.5, 6.7), 4),
        'conductivity': round(rng.uniform(1.6, 2.2), 4),
        'turbidity': round(rng.uniform(70, 100), 4),
        'orp': round(rng.uniform(105, 175), 4),
        'quality_label': 'Adulterant_C',
    })
    current_time += interval

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = pathlib.Path('data/raw/sample_sensor_data.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f'Saved {len(df)} rows to {out_path}')
print(df['quality_label'].value_counts())
print(df.head())
