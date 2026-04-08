import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and data
csv_path = '/Users/kuberapaul/Desktop/url detection 2/final_dataset_with_all_features_v3.1.csv'
model_path = '/Users/kuberapaul/Desktop/url detection 2/models/best_model.pkl'
output_dir = '/Users/kuberapaul/Desktop/url detection 2/models'

print("=" * 80)
print("DECISION TREE BASELINE - COMPREHENSIVE PERFORMANCE ANALYSIS")
print("=" * 80)

# Load data
print("\n[1] Loading Dataset & Model...")
df = pd.read_csv(csv_path)
X = df.drop(columns=['url', 'type', 'domain', 'scan_date', 'label'])
y = df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt_model = joblib.load(model_path)
y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)

print(f"    - Test set size: {X_test.shape[0]:,} samples")
print(f"    - Feature count: {X_test.shape[1]}")
print(f"    - Classes: {np.unique(y)}")

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Per-class metrics
precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n[2] Overall Performance Metrics")
print(f"    - Accuracy (weighted):  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"    - Precision (weighted): {precision_weighted:.4f}")
print(f"    - Recall (weighted):    {recall_weighted:.4f}")
print(f"    - F1-Score (weighted):  {f1_weighted:.4f}")

print("\n[3] Per-Class Performance")
class_names = ['Legitimate (0)', 'Phishing (1)', 'Malware (2)', 'Defacement (3)']
for idx, class_name in enumerate(class_names):
    print(f"\n    {class_name}:")
    print(f"      - Precision: {precision_per_class[idx]:.4f}")
    print(f"      - Recall:    {recall_per_class[idx]:.4f}")
    print(f"      - F1-Score:  {f1_per_class[idx]:.4f}")
    print(f"      - Support:   {(y_test == idx).sum():,} samples")

# Create comprehensive visualization
print("\n[4] Creating Comprehensive Performance Visualization...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Plot 1: Confusion Matrix (Large, top-left to middle)
ax1 = fig.add_subplot(gs[0:2, 0:2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax1,
            xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
ax1.set_title('Confusion Matrix - Decision Tree Baseline\n(130,238 Test Samples)', 
              fontsize=13, fontweight='bold', pad=15)
ax1.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

# Plot 2: Overall Metrics Bar Chart (top-right)
ax2 = fig.add_subplot(gs[0, 2])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision_weighted, recall_weighted, f1_weighted]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = ax2.barh(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xlim([0.8, 0.9])
ax2.set_title('Overall Metrics\n(Weighted Average)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Score', fontsize=10)
for i, (bar, val) in enumerate(zip(bars, values)):
    ax2.text(val + 0.002, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=9)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Per-Class F1 Scores (middle-right)
ax3 = fig.add_subplot(gs[1, 2])
colors_per_class = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = ax3.bar(range(4), f1_per_class, color=colors_per_class, edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(4))
ax3.set_xticklabels(['Legit', 'Phish', 'Malware', 'Defacement'], fontsize=9)
ax3.set_ylim([0.7, 0.95])
ax3.set_title('F1-Score by Class', fontsize=11, fontweight='bold')
ax3.set_ylabel('F1-Score', fontsize=10)
for i, (bar, val) in enumerate(zip(bars, f1_per_class)):
    ax3.text(i, val + 0.005, f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(y=f1_weighted, color='red', linestyle='--', linewidth=2, label=f'Weighted Avg: {f1_weighted:.4f}')
ax3.legend(fontsize=8)

# Plot 4: Precision, Recall, F1 by Class (bottom-left)
ax4 = fig.add_subplot(gs[2, 0:2])
x = np.arange(len(class_names))
width = 0.25
ax4.bar(x - width, precision_per_class, width, label='Precision', color='#2E86AB', edgecolor='black')
ax4.bar(x, recall_per_class, width, label='Recall', color='#A23B72', edgecolor='black')
ax4.bar(x + width, f1_per_class, width, label='F1-Score', color='#F18F01', edgecolor='black')
ax4.set_xlabel('Class', fontsize=11, fontweight='bold')
ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
ax4.set_title('Detailed Per-Class Metrics', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(['Legitimate', 'Phishing', 'Malware', 'Defacement'])
ax4.legend(fontsize=10)
ax4.set_ylim([0.6, 1.0])
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Class Distribution vs Prediction (bottom-right)
ax5 = fig.add_subplot(gs[2, 2])
true_counts = np.array([(y_test == i).sum() for i in range(4)])
pred_counts = np.array([(y_pred == i).sum() for i in range(4)])
x = np.arange(4)
width = 0.35
ax5.bar(x - width/2, true_counts, width, label='True', color='#2E86AB', alpha=0.8, edgecolor='black')
ax5.bar(x + width/2, pred_counts, width, label='Predicted', color='#F18F01', alpha=0.8, edgecolor='black')
ax5.set_xlabel('Class', fontsize=10, fontweight='bold')
ax5.set_ylabel('Count', fontsize=10, fontweight='bold')
ax5.set_title('True vs Predicted Distribution', fontsize=11, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(['Legit', 'Phish', 'Malware', 'Defacement'], fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3)

plt.suptitle('DECISION TREE BASELINE - COMPREHENSIVE PERFORMANCE DASHBOARD', 
             fontsize=15, fontweight='bold', y=0.995)

plt.savefig(f'{output_dir}/decision_tree_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: decision_tree_comprehensive_analysis.png")
plt.close()

# Create detailed performance summary
print("\n[5] Creating Performance Summary Report...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DECISION TREE BASELINE - PERFORMANCE SUMMARY', fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix (normalized)
ax = axes[0, 0]
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', cbar=True, ax=ax,
            xticklabels=class_names, yticklabels=class_names)
ax.set_title('Normalized Confusion Matrix\n(Row = % of True Class)', fontsize=11, fontweight='bold')
ax.set_ylabel('True Label', fontweight='bold')
ax.set_xlabel('Predicted Label', fontweight='bold')

# Plot 2: Metrics comparison
ax = axes[0, 1]
metrics_comparison = pd.DataFrame({
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class
}, index=[0, 1, 2, 3])
metrics_comparison.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72', '#F18F01'], width=0.8)
ax.set_title('Per-Class Performance Metrics', fontsize=11, fontweight='bold')
ax.set_xlabel('Class ID', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_xticklabels(['Legitimate (0)', 'Phishing (1)', 'Malware (2)', 'Defacement (3)'], rotation=45)
ax.legend(loc='best', fontsize=9)
ax.set_ylim([0.5, 1.0])
ax.grid(axis='y', alpha=0.3)

# Plot 3: Prediction accuracy per class
ax = axes[1, 0]
correct_per_class = np.diag(cm)
total_per_class = cm.sum(axis=1)
accuracy_per_class = correct_per_class / total_per_class
ax.bar(range(4), accuracy_per_class, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'], 
       edgecolor='black', linewidth=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels(['Legitimate', 'Phishing', 'Malware', 'Defacement'])
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Per-Class Accuracy (% Correctly Classified)', fontsize=11, fontweight='bold')
ax.set_ylim([0.7, 1.0])
for i, v in enumerate(accuracy_per_class):
    ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Plot 4: Error analysis
ax = axes[1, 1]
errors_per_class = total_per_class - correct_per_class
ax.bar(range(4), errors_per_class, color=['#E63946', '#E63946', '#E63946', '#E63946'],
       edgecolor='black', linewidth=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels(['Legitimate', 'Phishing', 'Malware', 'Defacement'])
ax.set_ylabel('Misclassified Count', fontweight='bold')
ax.set_title('Misclassification Count by Class', fontsize=11, fontweight='bold')
for i, v in enumerate(errors_per_class):
    ax.text(i, v + 100, f'{int(v):,}', ha='center', fontweight='bold', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/decision_tree_performance_summary.png', dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: decision_tree_performance_summary.png")
plt.close()

# Create classification report visualization
print("\n[6] Classification Report")
print("\n" + classification_report(y_test, y_pred, target_names=class_names, digits=4))

# Save detailed metrics to CSV
print("\n[7] Saving Detailed Metrics...")
metrics_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class,
    'Support': [(y_test == i).sum() for i in range(4)]
})
metrics_df.to_csv(f'{output_dir}/decision_tree_detailed_metrics.csv', index=False)
print(f"    ✓ Saved: decision_tree_detailed_metrics.csv")

# Summary statistics
print("\n[8] SUMMARY STATISTICS")
print(f"\n    ✓ Overall Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"    ✓ Weighted Precision:    {precision_weighted:.4f}")
print(f"    ✓ Weighted Recall:       {recall_weighted:.4f}")
print(f"    ✓ Weighted F1-Score:     {f1_weighted:.4f}")
print(f"\n    • Total Test Samples:    {len(y_test):,}")
print(f"    • Correct Predictions:   {(y_pred == y_test).sum():,}")
print(f"    • Wrong Predictions:     {(y_pred != y_test).sum():,}")
print(f"    • Error Rate:            {1-accuracy:.4f} ({(1-accuracy)*100:.2f}%)")

print("\n    CLASS-WISE PERFORMANCE:")
for idx, class_name in enumerate(class_names):
    correct = cm[idx, idx]
    total = cm[idx, :].sum()
    print(f"\n    {class_name}:")
    print(f"      - Accuracy:      {correct}/{total} = {correct/total:.1%}")
    print(f"      - F1-Score:      {f1_per_class[idx]:.4f}")
    print(f"      - Misclassified: {total - correct:,} samples")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
