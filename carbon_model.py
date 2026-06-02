# carbon_model.py
# Predicts soil organic carbon (SOC) from satellite indices using Random Forest
# Mirrors ML workflows used at Klim and Seqana

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# --- Load data ---
df = pd.read_csv('data/soil_carbon_data.csv')
print(f"Loaded {len(df)} samples")

# --- Features and target ---
# We use only the satellite indices as input (what a satellite can actually measure)
features = ['ndvi', 'evi', 'bsi']
X = df[features]
y = df['soc_percent']

# --- Train/test split ---
# 80% training, 20% testing — standard ML practice
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# --- Train Random Forest ---
model = RandomForestRegressor(
    n_estimators=100,   # 100 decision trees
    max_depth=10,       # prevents overfitting
    random_state=42
)
model.fit(X_train, y_train)
print("Model trained!")

# --- Evaluate ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"  MAE  : {mae:.3f}% SOC  (average prediction error)")
print(f"  R²   : {r2:.3f}       (1.0 = perfect, 0 = useless)")

# --- Feature importance ---
importances = model.feature_importances_
print(f"\nFeature Importance (which index matters most):")
for feat, imp in zip(features, importances):
    print(f"  {feat}: {imp:.3f}")

# --- Plot 1: Predicted vs Actual ---
os.makedirs('outputs', exist_ok=True)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', label='Perfect prediction')
plt.xlabel('Actual SOC (%)')
plt.ylabel('Predicted SOC (%)')
plt.title('Random Forest: Predicted vs Actual Soil Organic Carbon')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/predicted_vs_actual.png')
plt.show()

# --- Plot 2: Feature importance bar chart ---
plt.figure(figsize=(6, 4))
plt.bar(features, importances, color=['green', 'steelblue', 'brown'])
plt.xlabel('Satellite Index')
plt.ylabel('Importance Score')
plt.title('Which Satellite Index Best Predicts Soil Carbon?')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png')
plt.show()

print("\nCharts saved to outputs/")
