# generate_data.py
# Generates synthetic satellite index + soil organic carbon dataset

import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_SAMPLES = 500

ndvi = np.random.uniform(0.1, 0.85, NUM_SAMPLES)
evi = ndvi * np.random.uniform(0.7, 0.95, NUM_SAMPLES)
bsi = 1.0 - ndvi + np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
bsi = np.clip(bsi, 0, 1)

soc = (
    1.2 * ndvi +
    0.8 * evi -
    0.5 * bsi +
    np.random.normal(0, 0.15, NUM_SAMPLES) +
    0.5
)
soc = np.clip(soc, 0.3, 4.5)

land_use = np.random.choice(
    ['conventional', 'regenerative', 'degraded', 'fallow'],
    NUM_SAMPLES,
    p=[0.4, 0.3, 0.2, 0.1]
)

soc[land_use == 'regenerative'] += np.random.uniform(0.1, 0.4,
    np.sum(land_use == 'regenerative'))
soc[land_use == 'degraded'] -= np.random.uniform(0.1, 0.3,
    np.sum(land_use == 'degraded'))
soc = np.clip(soc, 0.3, 4.5)

df = pd.DataFrame({
    'ndvi': np.round(ndvi, 4),
    'evi': np.round(evi, 4),
    'bsi': np.round(bsi, 4),
    'land_use': land_use,
    'soc_percent': np.round(soc, 3)
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/soil_carbon_data.csv', index=False)

print(f"Dataset created: {len(df)} samples")
print(df.describe())
print("\nLand use distribution:")
print(df['land_use'].value_counts())