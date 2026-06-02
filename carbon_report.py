# carbon_report.py
# Generates a summary carbon report from model results
# Simulates MRV (Monitoring, Reporting, Verification) pipeline output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# --- Load data ---
df = pd.read_csv('data/soil_carbon_data.csv')
spatial = pd.read_csv('outputs/spatial_predictions.csv')

os.makedirs('outputs', exist_ok=True)

# --- Summary statistics by land use ---
print("=" * 50)
print("SOIL CARBON REPORT")
print("=" * 50)

summary = df.groupby('land_use')['soc_percent'].agg(['mean', 'std', 'count'])
summary.columns = ['Mean SOC %', 'Std Dev', 'Sample Count']
summary = summary.round(3)

print("\nSOC by Land Use Type:")
print(summary.to_string())

# Carbon stock estimate (simplified)
# Assumes 30cm depth, 1.3 g/cm³ bulk density, 1 hectare
# SOC stock (tonnes/ha) = SOC% * depth(m) * bulk density * 10000
DEPTH = 0.30
BULK_DENSITY = 1.3
AREA_HA = 1.0

summary['SOC Stock (t/ha)'] = (
    summary['Mean SOC %'] / 100 * DEPTH * BULK_DENSITY * 10000
).round(2)

print("\nEstimated SOC Stock (tonnes per hectare):")
print(summary[['Mean SOC %', 'SOC Stock (t/ha)']].to_string())

regen_stock = summary.loc['regenerative', 'SOC Stock (t/ha)']
conv_stock = summary.loc['conventional', 'SOC Stock (t/ha)']
difference = regen_stock - conv_stock

print(f"\nRegenerative vs Conventional farming:")
print(f"  Regenerative SOC stock : {regen_stock} t/ha")
print(f"  Conventional SOC stock : {conv_stock} t/ha")
print(f"  Carbon benefit         : +{difference:.2f} t/ha")

# --- Build report figure ---
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1: SOC distribution by land use
ax1 = fig.add_subplot(gs[0, 0])
land_use_types = summary.index.tolist()
colors = ['#e07b39', '#4caf50', '#c0392b', '#95a5a6']
means = summary['Mean SOC %'].values
stds = summary['Std Dev'].values
bars = ax1.bar(land_use_types, means, yerr=stds, color=colors,
               capsize=5, edgecolor='black', linewidth=0.8)
ax1.set_ylabel('Mean SOC (%)')
ax1.set_title('SOC by Land Use Type')
ax1.set_xlabel('Land Use')
for bar, val in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: SOC stock comparison
ax2 = fig.add_subplot(gs[0, 1])
stocks = summary['SOC Stock (t/ha)'].values
bars2 = ax2.bar(land_use_types, stocks, color=colors,
                edgecolor='black', linewidth=0.8)
ax2.set_ylabel('SOC Stock (tonnes/ha)')
ax2.set_title('Estimated Carbon Stock by Land Use')
ax2.set_xlabel('Land Use')
for bar, val in zip(bars2, stocks):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# Plot 3: SOC distribution histogram
ax3 = fig.add_subplot(gs[1, 0])
for lu, color in zip(land_use_types, colors):
    subset = df[df['land_use'] == lu]['soc_percent']
    ax3.hist(subset, bins=20, alpha=0.6, label=lu, color=color)
ax3.set_xlabel('SOC (%)')
ax3.set_ylabel('Frequency')
ax3.set_title('SOC Distribution by Land Use')
ax3.legend(fontsize=8)

# Plot 4: Spatial SOC map
ax4 = fig.add_subplot(gs[1, 1])
GRID_SIZE = 30
soc_grid = spatial['soc_predicted'].values.reshape(GRID_SIZE, GRID_SIZE)
im = ax4.imshow(soc_grid, cmap='YlOrBr', origin='lower')
plt.colorbar(im, ax=ax4, label='SOC (%)')
ax4.set_title('Spatial Carbon Map (30x30 km)')
ax4.set_xlabel('X (km)')
ax4.set_ylabel('Y (km)')

fig.suptitle('Soil Carbon MRV Report — soil-carbon-estimator',
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig('outputs/carbon_report.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nReport saved to outputs/carbon_report.png")

# --- Save text summary ---
with open('outputs/carbon_summary.txt', 'w') as f:
    f.write("SOIL CARBON MRV REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(summary.to_string())
    f.write(f"\n\nRegenerative vs Conventional benefit: +{difference:.2f} t/ha\n")
    f.write(f"Spatial mean SOC: {spatial['soc_predicted'].mean():.2f}%\n")

print("Text summary saved to outputs/carbon_summary.txt")
print("\nProject complete!")