import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
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
print("UNSUPERVISED LEARNING - ITERATION 1: ISOLATION FOREST BASELINE")
print("=" * 80)

# Load dataset
print("\n[1] Loading Dataset...")
df = pd.read_csv(csv_path)
print(f"    - Shape: {df.shape}")

# Identify target column
possible_targets = ['label', 'target', 'class', 'Label', 'Target', 'Class']
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    target_col = df.columns[-1]

print(f"    - Target column: {target_col}")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"    - Features shape: {X.shape}")
print(f"    - Target distribution:\n{y.value_counts().sort_index()}")

# Data Preprocessing
print("\n[2] Data Preprocessing...")
print(f"    - Missing values before: {X.isnull().sum().sum()}")

# Drop high-cardinality categorical columns
cols_to_drop = ['url', 'type', 'domain', 'scan_date']
cols_to_drop = [col for col in cols_to_drop if col in X.columns]
if cols_to_drop:
    print(f"    - Dropping high-cardinality columns: {cols_to_drop}")
    X = X.drop(columns=cols_to_drop)

# Handle categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"    - Categorical columns found: {categorical_cols}")
    for col in categorical_cols:
        X[col] = X[col].fillna('missing')
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"    - Shape after encoding: {X.shape}")

# Fill missing values in numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].mean())

print(f"    - Missing values after: {X.isnull().sum().sum()}")
print(f"    - Final features shape: {X.shape}")

# Feature scaling
print("\n[3] Feature Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("    - Scaling completed")

# ============================================================================
# UNSUPERVISED TRAINING: Train ONLY on Legitimate URLs (Class 0)
# ============================================================================
print("\n[4] Preparing Data for Unsupervised Learning...")
print(f"    - Original data shape: {X_scaled.shape}")

# Filter to legitimate URLs only (Class 0)
X_legitimate = X_scaled[y == 0]
print(f"    - Legitimate URLs only: {X_legitimate.shape}")
print(f"    - Percentage: {len(X_legitimate) / len(X_scaled) * 100:.2f}%")

# ============================================================================
# ITERATION 1: Basic Isolation Forest
# ============================================================================
print("\n[5] Training Isolation Forest (Iteration 1)...")
print("    Parameters:")
print("      - n_estimators: 100")
print("      - contamination: 0.35")
print("      - random_state: 42")

detector = IsolationForest(
    n_estimators=100,
    contamination=0.35,  # Expect ~35% anomalies (all attacks combined)
    random_state=42,
    n_jobs=-1
)

detector.fit(X_legitimate)
print("    - Model trained successfully")

# Make predictions on ALL data (including attacks)
print("\n[6] Making Predictions on All Data...")
anomaly_predictions = detector.predict(X_scaled)  # -1 = anomaly, +1 = normal
anomaly_scores = detector.score_samples(X_scaled)  # Continuous score

print(f"    - Predictions shape: {anomaly_predictions.shape}")
print(f"    - Predictions distribution:")
print(f"      Normal (+1):  {(anomaly_predictions == 1).sum():,}")
print(f"      Anomaly (-1): {(anomaly_predictions == -1).sum():,}")

# Convert predictions to binary (1 = anomaly, 0 = normal)
y_pred_binary = (anomaly_predictions == -1).astype(int)

# For comparison, create binary labels (0 = normal, 1+ = any attack)
y_binary = (y > 0).astype(int)

print(f"\n    - True labels (0=normal, 1=attack):")
print(f"      Normal (0): {(y_binary == 0).sum():,} ({(y_binary == 0).sum() / len(y_binary) * 100:.2f}%)")
print(f"      Attack (1): {(y_binary == 1).sum():,} ({(y_binary == 1).sum() / len(y_binary) * 100:.2f}%)")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n[7] Evaluation Metrics...")

# Calculate metrics treating it as binary classification
accuracy = accuracy_score(y_binary, y_pred_binary)
precision = precision_score(y_binary, y_pred_binary, zero_division=0)
recall = recall_score(y_binary, y_pred_binary, zero_division=0)
f1 = f1_score(y_binary, y_pred_binary, zero_division=0)

# ROC-AUC using anomaly scores
roc_auc = roc_auc_score(y_binary, -anomaly_scores)  # Negative because lower scores = more anomalous

print(f"\n    Binary Classification (Normal vs Any Attack):")
print(f"      Accuracy:  {accuracy:.4f}")
print(f"      Precision: {precision:.4f}")
print(f"      Recall:    {recall:.4f}")
print(f"      F1-Score:  {f1:.4f}")
print(f"      ROC-AUC:   {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_binary, y_pred_binary)
print(f"\n    Confusion Matrix:")
print(f"      True Negatives (Normal):    {cm[0, 0]:,}")
print(f"      False Positives (False Alarms): {cm[0, 1]:,}")
print(f"      False Negatives (Missed):   {cm[1, 0]:,}")
print(f"      True Positives (Detected):  {cm[1, 1]:,}")

# Detection rate by attack type
print(f"\n    Detection Rate by Attack Type:")
for attack_class in sorted(y.unique()):
    if attack_class != 0:  # Skip legitimate
        class_names = {1: "Phishing", 2: "Malware", 3: "Defacement"}
        class_name = class_names.get(attack_class, f"Class {attack_class}")
        
        mask = y == attack_class
        detected = (y_pred_binary[mask] == 1).sum()
        total = mask.sum()
        detection_rate = detected / total * 100 if total > 0 else 0
        
        print(f"      Class {attack_class} ({class_name:12}): {detected:,}/{total:,} = {detection_rate:6.2f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[8] Saving Results...")

# Save model
joblib.dump(detector, os.path.join(output_dir, 'unsupervised_iteration_1_if.pkl'))
print(f"    - Saved: unsupervised_iteration_1_if.pkl")

# Save scaler
joblib.dump(scaler, os.path.join(output_dir, 'unsupervised_scaler.pkl'))
print(f"    - Saved: unsupervised_scaler.pkl")

# Save results to CSV
results_df = pd.DataFrame({
    'True_Label': y.values,
    'Is_Attack': y_binary,
    'Predicted_Anomaly': y_pred_binary,
    'Anomaly_Score': anomaly_scores,
    'Anomaly_Prediction': anomaly_predictions
})

results_df.to_csv(os.path.join(output_dir, 'unsupervised_iteration_1_predictions.csv'), index=False)
print(f"    - Saved: unsupervised_iteration_1_predictions.csv")

# Save summary
summary_data = {
    'Iteration': ['1_Isolation_Forest'],
    'Algorithm': ['Isolation Forest'],
    'N_Estimators': [100],
    'Contamination': [0.35],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1_Score': [f1],
    'ROC_AUC': [roc_auc],
    'True_Negatives': [cm[0, 0]],
    'False_Positives': [cm[0, 1]],
    'False_Negatives': [cm[1, 0]],
    'True_Positives': [cm[1, 1]]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(output_dir, 'unsupervised_iterations_summary.csv'), index=False)
print(f"    - Saved: unsupervised_iterations_summary.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[9] Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Unsupervised Learning - Iteration 1: Isolation Forest', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
axes[0, 0].set_title('Confusion Matrix (Binary: Normal vs Attack)')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# Plot 2: Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [accuracy, precision, recall, f1, roc_auc]
colors = ['skyblue', 'lightgreen', 'salmon', 'plum', 'gold']

axes[0, 1].bar(metrics, values, color=colors)
axes[0, 1].set_title('Performance Metrics')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(values):
    axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)
plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Detection by Attack Type
attack_classes = sorted([c for c in y.unique() if c != 0])
class_names = {1: "Phishing", 2: "Malware", 3: "Defacement"}
detection_rates = []

for attack_class in attack_classes:
    mask = y == attack_class
    detected = (y_pred_binary[mask] == 1).sum()
    total = mask.sum()
    detection_rate = detected / total * 100 if total > 0 else 0
    detection_rates.append(detection_rate)

class_labels = [class_names.get(c, f"Class {c}") for c in attack_classes]
colors_attack = ['#FF6B6B', '#4ECDC4', '#45B7D1']
axes[1, 0].bar(class_labels, detection_rates, color=colors_attack)
axes[1, 0].set_title('Detection Rate by Attack Type')
axes[1, 0].set_ylabel('Detection Rate (%)')
axes[1, 0].set_ylim([0, 100])
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(detection_rates):
    axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)

# Plot 4: Anomaly Score Distribution
axes[1, 1].hist(anomaly_scores[y_binary == 0], bins=50, alpha=0.6, label='Normal (Class 0)', color='green')
axes[1, 1].hist(anomaly_scores[y_binary == 1], bins=50, alpha=0.6, label='Attack (Classes 1-3)', color='red')
axes[1, 1].set_title('Anomaly Score Distribution')
axes[1, 1].set_xlabel('Anomaly Score (lower = more anomalous)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'unsupervised_iteration_1_results.png'), dpi=300, bbox_inches='tight')
print(f"    - Saved: unsupervised_iteration_1_results.png")

plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 1 COMPLETED")
print("=" * 80)
print(f"\nAlgorithm: Isolation Forest")
print(f"Training Data: Legitimate URLs Only ({len(X_legitimate):,} samples)")
print(f"Test Data: All URLs ({len(X_scaled):,} samples)")
print(f"\nPerformance:")
print(f"  - Accuracy:  {accuracy:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall:    {recall:.4f}")
print(f"  - F1-Score:  {f1:.4f}")
print(f"  - ROC-AUC:   {roc_auc:.4f}")
print(f"\nKey Insights:")
print(f"  - Model trained on ONLY legitimate URLs (65.74% of dataset)")
print(f"  - Detects anomalies = anything different from 'normal'")
print(f"  - Expected: Lower detection than supervised (baseline for improvement)")
print(f"\nNext Steps:")
print(f"  - Iteration 2: Tune contamination parameter")
print(f"  - Iteration 3+: Add more detectors (LOF, Elliptic Envelope, etc.)")
print(f"  - Final: Build ensemble for 85-88% detection")
