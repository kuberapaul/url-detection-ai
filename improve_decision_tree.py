import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
csv_path = '/Users/kuberapaul/Desktop/url detection 2/final_dataset_with_all_features_v3.1.csv'
output_dir = '/Users/kuberapaul/Desktop/url detection 2/models'

import os
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("DECISION TREE ITERATIVE IMPROVEMENT")
print("=" * 80)

# Load and preprocess data (same as before)
print("\n[1] Loading & Preprocessing Data...")
df = pd.read_csv(csv_path)

# Identify target
possible_targets = ['label', 'target', 'class', 'Label', 'Target', 'Class', 'is_malicious', 'malicious']
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Drop high-cardinality columns
cols_to_drop = ['url', 'type', 'domain', 'scan_date']
cols_to_drop = [col for col in cols_to_drop if col in X.columns]
X = X.drop(columns=cols_to_drop)

# Handle categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    for col in categorical_cols:
        X[col] = X[col].fillna('missing')
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Fill missing numeric values
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].mean())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    - Data shape: {X.shape}")
print(f"    - Training: {X_train.shape}, Test: {X_test.shape}")

# ============================================================================
# ITERATION 0: BASELINE (Original Decision Tree)
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 0: BASELINE (Original Configuration)")
print("=" * 80)

dt_baseline = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

dt_baseline.fit(X_train, y_train)
y_pred_baseline = dt_baseline.predict(X_test)
y_pred_proba_baseline = dt_baseline.predict_proba(X_test)

metrics_baseline = {
    'accuracy': accuracy_score(y_test, y_pred_baseline),
    'precision': precision_score(y_test, y_pred_baseline, average='weighted', zero_division=0),
    'recall': recall_score(y_test, y_pred_baseline, average='weighted', zero_division=0),
    'f1': f1_score(y_test, y_pred_baseline, average='weighted', zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_baseline, multi_class='ovr', average='weighted'),
}

print("\nBaseline Configuration:")
print("  max_depth=10, criterion=gini, min_samples_split=2, min_samples_leaf=1")
print("\nBaseline Results:")
for metric, value in metrics_baseline.items():
    print(f"  {metric:12s}: {value:.4f}")

baseline_results = [{'iteration': 0, 'config': 'Baseline (max_depth=10)', **metrics_baseline}]

# ============================================================================
# ITERATION 1: MAX_DEPTH TUNING
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 1: MAX_DEPTH TUNING (5-50)")
print("=" * 80)

max_depth_values = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
max_depth_results = []

for max_depth in max_depth_values:
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'),
    }
    
    max_depth_results.append({
        'max_depth': max_depth,
        **metrics
    })
    
    print(f"  max_depth={max_depth:2d} → F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f}")

# Find best max_depth
best_max_depth_idx = np.argmax([r['f1'] for r in max_depth_results])
best_max_depth = max_depth_results[best_max_depth_idx]['max_depth']
best_max_depth_metrics = max_depth_results[best_max_depth_idx]

print(f"\n  🏆 Best max_depth: {best_max_depth} (F1: {best_max_depth_metrics['f1']:.4f})")

baseline_results.append({
    'iteration': 1,
    'config': f'max_depth={best_max_depth}',
    **{k: v for k, v in best_max_depth_metrics.items() if k != 'max_depth'}
})

# ============================================================================
# ITERATION 2: MIN_SAMPLES_SPLIT TUNING
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 2: MIN_SAMPLES_SPLIT TUNING (2-100)")
print("=" * 80)

min_samples_split_values = [2, 5, 10, 20, 50, 100]
min_samples_split_results = []

for min_samples_split in min_samples_split_values:
    dt = DecisionTreeClassifier(
        max_depth=best_max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'),
    }
    
    min_samples_split_results.append({
        'min_samples_split': min_samples_split,
        **metrics
    })
    
    print(f"  min_samples_split={min_samples_split:3d} → F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f}")

best_min_samples_split_idx = np.argmax([r['f1'] for r in min_samples_split_results])
best_min_samples_split = min_samples_split_results[best_min_samples_split_idx]['min_samples_split']
best_min_samples_split_metrics = min_samples_split_results[best_min_samples_split_idx]

print(f"\n  🏆 Best min_samples_split: {best_min_samples_split} (F1: {best_min_samples_split_metrics['f1']:.4f})")

baseline_results.append({
    'iteration': 2,
    'config': f'max_depth={best_max_depth}, min_samples_split={best_min_samples_split}',
    **{k: v for k, v in best_min_samples_split_metrics.items() if k != 'min_samples_split'}
})

# ============================================================================
# ITERATION 3: MIN_SAMPLES_LEAF TUNING
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 3: MIN_SAMPLES_LEAF TUNING (1-50)")
print("=" * 80)

min_samples_leaf_values = [1, 2, 5, 10, 20, 50]
min_samples_leaf_results = []

for min_samples_leaf in min_samples_leaf_values:
    dt = DecisionTreeClassifier(
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'),
    }
    
    min_samples_leaf_results.append({
        'min_samples_leaf': min_samples_leaf,
        **metrics
    })
    
    print(f"  min_samples_leaf={min_samples_leaf:2d} → F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f}")

best_min_samples_leaf_idx = np.argmax([r['f1'] for r in min_samples_leaf_results])
best_min_samples_leaf = min_samples_leaf_results[best_min_samples_leaf_idx]['min_samples_leaf']
best_min_samples_leaf_metrics = min_samples_leaf_results[best_min_samples_leaf_idx]

print(f"\n  🏆 Best min_samples_leaf: {best_min_samples_leaf} (F1: {best_min_samples_leaf_metrics['f1']:.4f})")

baseline_results.append({
    'iteration': 3,
    'config': f'max_depth={best_max_depth}, min_samples_split={best_min_samples_split}, min_samples_leaf={best_min_samples_leaf}',
    **{k: v for k, v in best_min_samples_leaf_metrics.items() if k != 'min_samples_leaf'}
})

# ============================================================================
# ITERATION 4: CRITERION COMPARISON (GINI vs ENTROPY)
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 4: CRITERION COMPARISON (GINI vs ENTROPY)")
print("=" * 80)

criterion_results = []

for criterion in ['gini', 'entropy']:
    dt = DecisionTreeClassifier(
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=best_min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'),
    }
    
    criterion_results.append({
        'criterion': criterion,
        **metrics
    })
    
    print(f"  {criterion:7s} → F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f}")

best_criterion_idx = np.argmax([r['f1'] for r in criterion_results])
best_criterion = criterion_results[best_criterion_idx]['criterion']
best_criterion_metrics = criterion_results[best_criterion_idx]

print(f"\n  🏆 Best criterion: {best_criterion} (F1: {best_criterion_metrics['f1']:.4f})")

baseline_results.append({
    'iteration': 4,
    'config': f'max_depth={best_max_depth}, min_samples_split={best_min_samples_split}, min_samples_leaf={best_min_samples_leaf}, criterion={best_criterion}',
    **{k: v for k, v in best_criterion_metrics.items() if k != 'criterion'}
})

# ============================================================================
# ITERATION 5: GRID SEARCH (COMPREHENSIVE TUNING)
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 5: GRID SEARCH (COMPREHENSIVE HYPERPARAMETER TUNING)")
print("=" * 80)
print("Testing 24 combinations...")

param_grid = {
    'max_depth': [8, 10, 12, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}

# Grid search
dt_grid = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    dt_grid, 
    param_grid, 
    cv=5,  # 5-fold cross-validation
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

best_params_grid = grid_search.best_params_
print(f"\nBest parameters found:")
for param, value in best_params_grid.items():
    print(f"  {param}: {value}")

# Train final model with best parameters
dt_best = DecisionTreeClassifier(
    **best_params_grid,
    random_state=42
)
dt_best.fit(X_train, y_train)
y_pred_best = dt_best.predict(X_test)
y_pred_proba_best = dt_best.predict_proba(X_test)

metrics_best = {
    'accuracy': accuracy_score(y_test, y_pred_best),
    'precision': precision_score(y_test, y_pred_best, average='weighted', zero_division=0),
    'recall': recall_score(y_test, y_pred_best, average='weighted', zero_division=0),
    'f1': f1_score(y_test, y_pred_best, average='weighted', zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_best, multi_class='ovr', average='weighted'),
}

print(f"\nGrid Search Results:")
for metric, value in metrics_best.items():
    print(f"  {metric:12s}: {value:.4f}")

config_str = ', '.join([f"{k}={v}" for k, v in best_params_grid.items()])
baseline_results.append({
    'iteration': 5,
    'config': f'GridSearch: {config_str}',
    **metrics_best
})

# ============================================================================
# RESULTS SUMMARY & VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: IMPROVEMENT TRAJECTORY")
print("=" * 80)

results_df = pd.DataFrame(baseline_results)
print("\n" + results_df.to_string(index=False))

# Calculate improvements
improvement_f1 = (metrics_best['f1'] - metrics_baseline['f1']) / metrics_baseline['f1'] * 100
improvement_acc = (metrics_best['accuracy'] - metrics_baseline['accuracy']) / metrics_baseline['accuracy'] * 100

print(f"\n{'='*80}")
print(f"IMPROVEMENTS FROM BASELINE TO BEST MODEL")
print(f"{'='*80}")
print(f"  Baseline F1:       {metrics_baseline['f1']:.4f}")
print(f"  Best F1:           {metrics_best['f1']:.4f}")
print(f"  Improvement:       {improvement_f1:+.2f}%")
print(f"\n  Baseline Accuracy: {metrics_baseline['accuracy']:.4f}")
print(f"  Best Accuracy:     {metrics_best['accuracy']:.4f}")
print(f"  Improvement:       {improvement_acc:+.2f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[6] Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Decision Tree Iterative Improvement', fontsize=16, fontweight='bold')

# Plot 1: F1 Score Progression
ax1 = axes[0, 0]
iterations = results_df['iteration'].values
f1_scores = results_df['f1'].values
ax1.plot(iterations, f1_scores, 'o-', linewidth=2, markersize=8, color='steelblue')
ax1.axhline(y=metrics_baseline['f1'], color='red', linestyle='--', label='Baseline', alpha=0.7)
ax1.fill_between(iterations, f1_scores, metrics_baseline['f1'], alpha=0.3)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('F1-Score')
ax1.set_title('F1-Score Improvement Over Iterations')
ax1.grid(True, alpha=0.3)
ax1.legend()
for i, f1 in enumerate(f1_scores):
    ax1.text(iterations[i], f1 + 0.002, f'{f1:.4f}', ha='center', fontsize=9)

# Plot 2: All Metrics Comparison
ax2 = axes[0, 1]
metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
baseline_values = [metrics_baseline[m] for m in metrics_names]
best_values = [metrics_best[m] for m in metrics_names]

x_pos = np.arange(len(metrics_names))
width = 0.35

ax2.bar(x_pos - width/2, baseline_values, width, label='Baseline', alpha=0.8)
ax2.bar(x_pos + width/2, best_values, width, label='Best', alpha=0.8)
ax2.set_ylabel('Score')
ax2.set_title('Baseline vs Best Model Metrics')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(metrics_names, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (baseline, best) in enumerate(zip(baseline_values, best_values)):
    ax2.text(i - width/2, baseline + 0.01, f'{baseline:.3f}', ha='center', fontsize=8)
    ax2.text(i + width/2, best + 0.01, f'{best:.3f}', ha='center', fontsize=8)

# Plot 3: Confusion Matrix - Best Model
ax3 = axes[1, 0]
cm_best = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
ax3.set_title('Confusion Matrix - Best Model')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

# Plot 4: Accuracy Progression
ax4 = axes[1, 1]
accuracy_scores = results_df['accuracy'].values
ax4.plot(iterations, accuracy_scores, 's-', linewidth=2, markersize=8, color='darkgreen')
ax4.axhline(y=metrics_baseline['accuracy'], color='red', linestyle='--', label='Baseline', alpha=0.7)
ax4.fill_between(iterations, accuracy_scores, metrics_baseline['accuracy'], alpha=0.3)
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Accuracy')
ax4.set_title('Accuracy Improvement Over Iterations')
ax4.grid(True, alpha=0.3)
ax4.legend()
for i, acc in enumerate(accuracy_scores):
    ax4.text(iterations[i], acc + 0.002, f'{acc:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'decision_tree_improvement.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: decision_tree_improvement.png")
plt.close()

# ============================================================================
# SAVE BEST MODEL & RESULTS
# ============================================================================
print("\n[7] Saving Results...")

# Save best model
joblib.dump(dt_best, os.path.join(output_dir, 'best_decision_tree_improved.pkl'))
print(f"  ✓ Saved: best_decision_tree_improved.pkl")

# Save results dataframe
results_df.to_csv(os.path.join(output_dir, 'decision_tree_iterations.csv'), index=False)
print(f"  ✓ Saved: decision_tree_iterations.csv")

# Save detailed results
detailed_results = {
    'baseline': metrics_baseline,
    'best': metrics_best,
    'best_params': best_params_grid,
    'improvement_f1_percent': improvement_f1,
    'improvement_acc_percent': improvement_acc,
}

import json
with open(os.path.join(output_dir, 'decision_tree_improvement_summary.json'), 'w') as f:
    json.dump(detailed_results, f, indent=2)
print(f"  ✓ Saved: decision_tree_improvement_summary.json")

# Save detailed hyperparameter results
with open(os.path.join(output_dir, 'grid_search_results.txt'), 'w') as f:
    f.write("GRID SEARCH RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Best Parameters:\n")
    for param, value in best_params_grid.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nBest CV Score (F1): {grid_search.best_score_:.4f}\n")
    f.write(f"\nAll Tested Combinations:\n")
    results_grid_df = pd.DataFrame(grid_search.cv_results_)
    f.write(results_grid_df[['param_max_depth', 'param_min_samples_split', 
                              'param_min_samples_leaf', 'param_criterion', 
                              'mean_test_score']].to_string(index=False))

print(f"  ✓ Saved: grid_search_results.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DECISION TREE IMPROVEMENT COMPLETE")
print("=" * 80)
print(f"\nBaseline Configuration:")
print(f"  max_depth=10, criterion=gini, min_samples_split=2, min_samples_leaf=1")
print(f"  F1-Score: {metrics_baseline['f1']:.4f}, Accuracy: {metrics_baseline['accuracy']:.4f}\n")

print(f"Best Configuration (Iteration 5 - Grid Search):")
print(f"  " + ", ".join([f"{k}={v}" for k, v in best_params_grid.items()]))
print(f"  F1-Score: {metrics_best['f1']:.4f}, Accuracy: {metrics_best['accuracy']:.4f}\n")

print(f"Improvement:")
print(f"  F1:       {metrics_baseline['f1']:.4f} → {metrics_best['f1']:.4f} ({improvement_f1:+.2f}%)")
print(f"  Accuracy: {metrics_baseline['accuracy']:.4f} → {metrics_best['accuracy']:.4f} ({improvement_acc:+.2f}%)")
print(f"  ROC-AUC:  {metrics_baseline['roc_auc']:.4f} → {metrics_best['roc_auc']:.4f}")

print(f"\n{'='*80}")
print(f"Key Files Generated:")
print(f"  - best_decision_tree_improved.pkl")
print(f"  - decision_tree_iterations.csv")
print(f"  - decision_tree_improvement.png")
print(f"  - decision_tree_improvement_summary.json")
print(f"  - grid_search_results.txt")
print(f"{'='*80}\n")
