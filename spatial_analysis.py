# spatial_analysis.py
# Simulates a spatial grid of satellite observations and maps predicted SOC
# Mirrors carbon mapping workflows at Nala Earth and Orbio Earth

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# --- Retrain model on full dataset ---
df = pd.read_csv('data/soil_carbon_data.csv')
features = ['ndvi', 'evi', 'bsi']
X = df[features]
y = df['soc_percent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
print("Model retrained for spatial prediction")

# --- Create a 30x30 spatial grid ---
# Think of this as a 30x30 km area with one satellite pixel per km
GRID_SIZE = 30
print(f"Creating {GRID_SIZE}x{GRID_SIZE} spatial grid...")

np.random.seed(7)

# Simulate spatial variation — fields cluster together
# We use a gradient to mimic realistic landscape patterns
x_coords = np.linspace(0, 1, GRID_SIZE)
y_coords = np.linspace(0, 1, GRID_SIZE)
xx, yy = np.meshgrid(x_coords, y_coords)

# NDVI varies across landscape (higher in wetter/northern areas)
ndvi_grid = 0.3 + 0.4 * yy + 0.15 * np.sin(xx * np.pi * 3) + \
            np.random.normal(0, 0.05, (GRID_SIZE, GRID_SIZE))
ndvi_grid = np.clip(ndvi_grid, 0.1, 0.85)

# EVI follows NDVI with slight variation
evi_grid = ndvi_grid * (0.8 + np.random.normal(0, 0.05, (GRID_SIZE, GRID_SIZE)))
evi_grid = np.clip(evi_grid, 0.05, 0.85)

# BSI inversely related to NDVI
bsi_grid = 1.0 - ndvi_grid + np.random.normal(0, 0.05, (GRID_SIZE, GRID_SIZE))
bsi_grid = np.clip(bsi_grid, 0, 1)

# --- Flatten grid for prediction ---
ndvi_flat = ndvi_grid.flatten()
evi_flat = evi_grid.flatten()
bsi_flat = bsi_grid.flatten()

grid_df = pd.DataFrame({
    'ndvi': ndvi_flat,
    'evi': evi_flat,
    'bsi': bsi_flat
})

# --- Predict SOC for every pixel ---
soc_predicted = model.predict(grid_df)
soc_grid = soc_predicted.reshape(GRID_SIZE, GRID_SIZE)

print(f"SOC predictions complete")
print(f"  Min SOC: {soc_grid.min():.2f}%")
print(f"  Max SOC: {soc_grid.max():.2f}%")
print(f"  Mean SOC: {soc_grid.mean():.2f}%")

# --- Save spatial results ---
os.makedirs('outputs', exist_ok=True)
grid_df['soc_predicted'] = soc_predicted
grid_df['x'] = [i % GRID_SIZE for i in range(GRID_SIZE * GRID_SIZE)]
grid_df['y'] = [i // GRID_SIZE for i in range(GRID_SIZE * GRID_SIZE)]
grid_df.to_csv('outputs/spatial_predictions.csv', index=False)

# --- Plot 1: SOC carbon map ---
plt.figure(figsize=(8, 7))
im = plt.imshow(soc_grid, cmap='YlOrBr', origin='lower',
                vmin=soc_grid.min(), vmax=soc_grid.max())
plt.colorbar(im, label='Predicted SOC (%)')
plt.title('Predicted Soil Organic Carbon Map\n(30x30 km landscape)')
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.tight_layout()
plt.savefig('outputs/soc_carbon_map.png', dpi=150)
plt.show()

# --- Plot 2: NDVI map side by side with SOC ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes[0].imshow(ndvi_grid, cmap='Greens', origin='lower')
fig.colorbar(im1, ax=axes[0], label='NDVI')
axes[0].set_title('NDVI (Satellite Input)')
axes[0].set_xlabel('X (km)')
axes[0].set_ylabel('Y (km)')

im2 = axes[1].imshow(soc_grid, cmap='YlOrBr', origin='lower')
fig.colorbar(im2, ax=axes[1], label='Predicted SOC (%)')
axes[1].set_title('Predicted Soil Carbon (Model Output)')
axes[1].set_xlabel('X (km)')

plt.suptitle('From Satellite Signal to Carbon Map', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/ndvi_vs_soc_map.png', dpi=150)
plt.show()

print("\nMaps saved to outputs/")