import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# F1 scores data
# From train_model.py (Claude Haiku 4.5)
claude_models = {
    'Logistic Regression': 0.8401,
    'Decision Tree': 0.8487,
    'Random Forest': 0.8465,
    'Gradient Boosting': 0.8412
}

# From gpt_first_model.py (GPT 5.1)
gpt_f1 = 0.8341

# Create visualization
fig, ax = plt.subplots(figsize=(12, 7))

# Prepare data for plotting
models = list(claude_models.keys()) + ['GPT Decision Tree']
f1_scores = list(claude_models.values()) + [gpt_f1]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#E63946']  # Different color for GPT

# Create bar chart
bars = ax.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on top of bars
for bar, score in zip(bars, f1_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight best model (Claude's Decision Tree)
best_idx = f1_scores.index(max(claude_models.values()))
bars[best_idx].set_linewidth(3)
bars[best_idx].set_edgecolor('gold')

# Add horizontal line for GPT baseline
ax.axhline(y=gpt_f1, color='#E63946', linestyle='--', linewidth=2, alpha=0.7, label='GPT Baseline')

# Customize chart
ax.set_ylabel('F1-Score (Weighted)', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title('F1-Score Comparison: Claude vs GPT Initial Code\n(Primary Evaluation Metric)', 
             fontsize=14, fontweight='bold', pad=20)

# Set y-axis limits to show more detail
ax.set_ylim([0.82, 0.86])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Add legend
ax.text(0.02, 0.98, 'Gold Border = Best Claude Model', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add statistics box
stats_text = f"""Model Comparison Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━
Claude Best: Decision Tree (0.8487)
Claude Mean: {np.mean(list(claude_models.values())):.4f}
GPT Baseline: {gpt_f1:.4f}
Difference: {max(claude_models.values()) - gpt_f1:.4f} (+{((max(claude_models.values()) - gpt_f1)/gpt_f1)*100:.2f}%)"""

ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10, 
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        family='monospace')

plt.tight_layout()
plt.savefig('/Users/kuberapaul/Desktop/url detection 2/models/f1_comparison_claude_vs_gpt.png', 
            dpi=300, bbox_inches='tight')
print("✓ Graph saved: models/f1_comparison_claude_vs_gpt.png")

# Also create a detailed metrics table
comparison_data = {
    'Model': list(claude_models.keys()) + ['GPT Decision Tree'],
    'F1-Score': list(claude_models.values()) + [gpt_f1],
    'Source': ['Claude Haiku 4.5'] * 4 + ['GPT 5.1'],
    'Ranking': [2, 1, 3, 4, 5]
}

df = pd.DataFrame(comparison_data)
df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
df['Ranking'] = range(1, len(df) + 1)

print("\n" + "="*60)
print("F1-SCORE COMPARISON TABLE")
print("="*60)
print(df.to_string(index=False))
print("="*60)

# Save to CSV
df.to_csv('/Users/kuberapaul/Desktop/url detection 2/models/f1_comparison_table.csv', index=False)
print("✓ Table saved: models/f1_comparison_table.csv")

print(f"\n📊 Analysis:")
print(f"   • Claude's best model (Decision Tree): {max(claude_models.values()):.4f}")
print(f"   • GPT baseline (Decision Tree): {gpt_f1:.4f}")
print(f"   • Improvement: +{max(claude_models.values()) - gpt_f1:.4f} ({((max(claude_models.values()) - gpt_f1)/gpt_f1)*100:.2f}%)")
print(f"   • Claude explores 4 models; GPT uses only 1 conservative model")
