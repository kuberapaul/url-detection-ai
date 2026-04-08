"""
Generate all 5 required figures for assessment report (Section 7)
Run after improve_decision_tree.py to get all results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
print("Loading data and models...")
df = pd.read_csv('/Users/kuberapaul/Desktop/url detection 2/final_dataset_with_all_features_v3.1.csv')
best_model = joblib.load('/Users/kuberapaul/Desktop/url detection 2/models/best_model.pkl')
scaler = joblib.load('/Users/kuberapaul/Desktop/url detection 2/models/scaler.pkl')

# Preprocessing (same as train_model.py)
df_processed = df.copy()
columns_to_drop = ['url', 'type', 'domain', 'scan_date']
df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))

# Separate features and target
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
if 'label' in categorical_cols:
    categorical_cols.remove('label')

X = df_processed.drop('label', axis=1)
y = df_processed['label']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Apply test split (use same split as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Generate predictions
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)

# Get iteration results if available
try:
    iterations_df = pd.read_csv('/Users/kuberapaul/Desktop/url detection 2/models/decision_tree_iterations.csv')
    has_iterations = True
except:
    has_iterations = False
    print("Note: decision_tree_iterations.csv not found. Skipping Figure 3.")

# Class names
class_names = ['Legitimate', 'Phishing', 'Malware', 'Defacement']

# ============================================================================
# FIGURE 1: Class Distribution (Imbalance)
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
class_counts = y.value_counts().sort_index()
percentages = (class_counts / len(y)) * 100

colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
bars = ax1.bar(range(len(class_counts)), class_counts.values, color=colors, alpha=0.7, edgecolor='black')

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, percentages)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.1f}%\n({int(height):,})',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.set_xlabel('URL Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of URLs', fontsize=12, fontweight='bold')
ax1.set_title('Figure 1: Dataset Class Distribution (Imbalanced)', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(class_counts)))
ax1.set_xticklabels(class_names)
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/kuberapaul/Desktop/url detection 2/models/figure_1_class_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Figure 1: Class distribution saved")
plt.close()

# ============================================================================
# FIGURE 2: Confusion Matrix (Best Model)
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)

# Normalize for percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

sns.heatmap(cm, annot=cm_percent, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Count'},
            xticklabels=class_names, yticklabels=class_names, ax=ax2,
            annot_kws={'size': 11, 'weight': 'bold'}, linewidths=1, linecolor='black')

ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax2.set_title('Figure 2: Confusion Matrix - Best Model (Percentage %)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/kuberapaul/Desktop/url detection 2/models/figure_2_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Figure 2: Confusion matrix saved")
plt.close()

# ============================================================================
# FIGURE 3: Iteration Progress (F1-Score Improvement)
# ============================================================================
if has_iterations:
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    iterations_df_sorted = iterations_df.sort_values('iteration')
    ax3.plot(iterations_df_sorted['iteration'], iterations_df_sorted['f1_score'], 
             marker='o', linewidth=2.5, markersize=10, color='#e74c3c', label='F1-Score')
    
    # Add value labels on points
    for idx, row in iterations_df_sorted.iterrows():
        ax3.text(row['iteration'], row['f1_score'] + 0.002, f"{row['f1_score']:.4f}",
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1-Score (Weighted)', fontsize=12, fontweight='bold')
    ax3.set_title('Figure 3: Iterative Improvement - Decision Tree Hyperparameter Tuning', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([iterations_df['f1_score'].min() - 0.01, iterations_df['f1_score'].max() + 0.02])
    ax3.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.savefig('/Users/kuberapaul/Desktop/url detection 2/models/figure_3_iteration_progress.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 3: Iteration progress saved")
    plt.close()

# ============================================================================
# FIGURE 4: Metrics Comparison (Best Model)
# ============================================================================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# ROC-AUC (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')

metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'ROC-AUC': roc_auc
}

fig4, ax4 = plt.subplots(figsize=(10, 6))
metrics_names = list(metrics_dict.keys())
metrics_values = list(metrics_dict.values())

colors_metrics = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
bars = ax4.bar(metrics_names, metrics_values, color=colors_metrics, alpha=0.7, edgecolor='black')

# Add value labels on bars
for bar, val in zip(bars, metrics_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
ax4.set_title('Figure 4: Best Model Performance Metrics', fontsize=14, fontweight='bold')
ax4.set_ylim([0, 1.0])
ax4.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/kuberapaul/Desktop/url detection 2/models/figure_4_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Figure 4: Metrics comparison saved")
plt.close()

# ============================================================================
# FIGURE 5: Baseline vs Best Model Comparison
# ============================================================================
# Baseline (from train_model.py results)
baseline_results = {
    'Accuracy': 0.8727,
    'Precision': 0.8532,
    'Recall': 0.8727,
    'F1-Score': 0.8487,
    'ROC-AUC': 0.9417
}

best_results = metrics_dict

fig5, ax5 = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax5.bar(x - width/2, list(baseline_results.values()), width, label='Baseline (87.27%)',
                color='#95a5a6', alpha=0.8, edgecolor='black')
bars2 = ax5.bar(x + width/2, metrics_values, width, label='Best Iteration',
                color='#e74c3c', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
ax5.set_title('Figure 5: Baseline vs Best Iteration - Performance Improvement', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_names)
ax5.set_ylim([0, 1.0])
ax5.legend(fontsize=11, loc='lower right')
ax5.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/kuberapaul/Desktop/url detection 2/models/figure_5_baseline_vs_best.png', dpi=300, bbox_inches='tight')
print("✓ Figure 5: Baseline vs best comparison saved")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*60)
print("REPORT SUMMARY - ALL METRICS")
print("="*60)
print(f"\nBaseline Model (Initial Code):")
print(f"  Accuracy:  {baseline_results['Accuracy']:.4f} (87.27%)")
print(f"  F1-Score:  {baseline_results['F1-Score']:.4f} (84.87%)")
print(f"  ROC-AUC:   {baseline_results['ROC-AUC']:.4f}")

print(f"\nBest Model (After Iterations):")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

improvement_f1 = (f1 - baseline_results['F1-Score']) * 100
improvement_accuracy = (accuracy - baseline_results['Accuracy']) * 100

print(f"\nImprovement:")
print(f"  F1-Score:  +{improvement_f1:.2f}% (target was +0.5-1.5%)")
print(f"  Accuracy:  +{improvement_accuracy:.2f}%")

print(f"\n✓ All 5 figures saved to /models/")
print(f"✓ Ready for report Section 7 (Figures)")
print("="*60)
