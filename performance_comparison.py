# create_performance_comparison.py
"""
Create performance comparison visualizations for presentation
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create performance data
data = {
    'Model': [
        'Content-Based\n(Genre)',
        'Content-Based\n(TF-IDF)',
        'SVD',
        'NMF',
        'KNN',
        'Neural CF',
        'Hybrid\n(SVD+NMF)'
    ],
    'MAE': [None, None, 3.29, None, None, 0.65, 2.44],
    'RMSE': [None, None, 3.46, None, None, 0.71, 2.84],
    'Hit_Rate': [0.0306, 0.0309, None, None, None, None, None],
    'Category': [
        'Content-Based',
        'Content-Based',
        'Collaborative',
        'Collaborative',
        'Collaborative',
        'Collaborative',
        'Hybrid'
    ]
}

df = pd.DataFrame(data)

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: MAE Comparison
mae_data = df[df['MAE'].notna()]
colors = ['#FF6B6B' if x == 'Hybrid' else '#4ECDC4' if x == 'Collaborative' else '#95E1D3' 
          for x in mae_data['Category']]
axes[0].barh(mae_data['Model'], mae_data['MAE'], color=colors, edgecolor='black')
axes[0].set_xlabel('MAE (Lower is Better)', fontsize=12, fontweight='bold')
axes[0].set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Add values
for i, v in enumerate(mae_data['MAE']):
    axes[0].text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold')

# Plot 2: RMSE Comparison
rmse_data = df[df['RMSE'].notna()]
colors = ['#FF6B6B' if x == 'Hybrid' else '#4ECDC4' if x == 'Collaborative' else '#95E1D3' 
          for x in rmse_data['Category']]
axes[1].barh(rmse_data['Model'], rmse_data['RMSE'], color=colors, edgecolor='black')
axes[1].set_xlabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
axes[1].set_title('Root Mean Squared Error Comparison', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

# Add values
for i, v in enumerate(rmse_data['RMSE']):
    axes[1].text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold')

# Plot 3: Hit Rate for Content-Based
hit_data = df[df['Hit_Rate'].notna()]
colors = ['#95E1D3' for _ in hit_data['Category']]
axes[2].barh(hit_data['Model'], hit_data['Hit_Rate'] * 100, color=colors, edgecolor='black')
axes[2].set_xlabel('Hit Rate (%)', fontsize=12, fontweight='bold')
axes[2].set_title('Content-Based Hit Rate', fontsize=14, fontweight='bold')
axes[2].invert_yaxis()
axes[2].grid(axis='x', alpha=0.3)

# Add values
for i, v in enumerate(hit_data['Hit_Rate']):
    axes[2].text(v * 100 + 0.1, i, f'{v*100:.2f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('reports/eda_figures/07_final_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: reports/eda_figures/07_final_model_comparison.png")

# Create summary table
print("\n📊 FINAL PERFORMANCE SUMMARY")
print("="*70)
print(df.to_string(index=False))

# Save to CSV
df.to_csv('reports/final_model_comparison.csv', index=False)
print("\n✅ Saved: reports/final_model_comparison.csv")
