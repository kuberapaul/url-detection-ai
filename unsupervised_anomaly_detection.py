import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
csv_path = '/Users/kuberapaul/Desktop/url detection 2/final_dataset_with_all_features_v3.1.csv'
output_dir = '/Users/kuberapaul/Desktop/url detection 2/models'

import os
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("UNSUPERVISED ANOMALY DETECTION FOR ZERO-DAY THREATS")
print("=" * 80)

# Load dataset
print("\n[1] Loading & Preprocessing Data...")
df = pd.read_csv(csv_path)

# Identify target and features
possible_targets = ['label', 'target', 'class', 'Label', 'Target', 'Class', 'is_malicious', 'malicious']
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# Drop high-cardinality columns
cols_to_drop = ['url', 'type', 'domain', 'scan_date']
cols_to_drop = [col for col in cols_to_drop if col in X.columns]
if cols_to_drop:
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

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    - Data shape: {X.shape}")
print(f"    - Training: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"    - Features scaled to mean=0, std=1")

# ============================================================================
# ITERATION 0: ISOLATION FOREST (Baseline Unsupervised)
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 0: ISOLATION FOREST (Baseline Unsupervised)")
print("=" * 80)

iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(X_train_scaled)
iso_pred = iso_forest.predict(X_test_scaled)
iso_scores = iso_forest.score_samples(X_test_scaled)

iso_anomalies = np.sum(iso_pred == -1)
iso_normal = np.sum(iso_pred == 1)

print("\nIsolation Forest Results:")
print(f"    - Normal samples (value=1): {iso_normal}")
print(f"    - Anomalies (value=-1): {iso_anomalies}")
print(f"    - Anomaly score range: [{iso_scores.min():.4f}, {iso_scores.max():.4f}]")
print(f"    - Mean anomaly score (normal): {iso_scores[iso_pred == 1].mean():.4f}")
print(f"    - Mean anomaly score (anomaly): {iso_scores[iso_pred == -1].mean():.4f}")

joblib.dump(iso_forest, os.path.join(output_dir, 'isolation_forest.pkl'))
print("\n    ✓ Saved: isolation_forest.pkl")

# ============================================================================
# ITERATION 1: AUTOENCODER (Deep Learning)
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 1: AUTOENCODER (Deep Learning)")
print("=" * 80)

print("\nAutoencoder Architecture:")
print(f"    - Input: {X_train_scaled.shape[1]} features")
print(f"    - Encoder: {X_train_scaled.shape[1]} → 32 → 16 → 8 (bottleneck)")
print(f"    - Decoder: 8 → 16 → 32 → {X_train_scaled.shape[1]}")
print(f"    - Loss: Mean Squared Error (MSE)")

# Build autoencoder
encoder_input = keras.Input(shape=(X_train_scaled.shape[1],))
encoded = layers.Dense(32, activation='relu')(encoder_input)
encoded = layers.Dense(16, activation='relu')(encoded)
bottleneck = layers.Dense(8, activation='relu')(encoded)
decoded = layers.Dense(16, activation='relu')(bottleneck)
decoded = layers.Dense(32, activation='relu')(decoded)
decoder_output = layers.Dense(X_train_scaled.shape[1], activation='sigmoid')(decoded)

autoencoder = keras.Model(encoder_input, decoder_output)
autoencoder.compile(optimizer='adam', loss='mse')

print("\n    Training...")
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=0
)

print(f"    ✓ Training completed ({len(history.history['loss'])} epochs)")

# Calculate reconstruction error (MAE)
train_mae = np.mean(np.abs(autoencoder.predict(X_train_scaled, verbose=0) - X_train_scaled), axis=1)
test_mae = np.mean(np.abs(autoencoder.predict(X_test_scaled, verbose=0) - X_test_scaled), axis=1)

# Threshold at 95th percentile of training error
threshold = np.percentile(train_mae, 95)
ae_pred = np.where(test_mae > threshold, -1, 1)
ae_anomalies = np.sum(ae_pred == -1)
ae_normal = np.sum(ae_pred == 1)

print("\nAutoencoder Results:")
print(f"    - Training MAE: mean={train_mae.mean():.4f}, std={train_mae.std():.4f}")
print(f"    - Test MAE: mean={test_mae.mean():.4f}, std={test_mae.std():.4f}")
print(f"    - Anomaly threshold (95th percentile): {threshold:.4f}")
print(f"    - Anomalies detected: {ae_anomalies}")
print(f"    - Normal samples: {ae_normal}")

autoencoder.save(os.path.join(output_dir, 'autoencoder_model.h5'))
print("\n    ✓ Saved: autoencoder_model.h5")

# ============================================================================
# ITERATION 2: LOCAL OUTLIER FACTOR (Density-Based)
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION 2: LOCAL OUTLIER FACTOR (Density-Based)")
print("=" * 80)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(X_train_scaled)
lof_pred = lof.predict(X_test_scaled)
lof_scores = lof.score_samples(X_test_scaled)

lof_anomalies = np.sum(lof_pred == -1)
lof_normal = np.sum(lof_pred == 1)

print("\nLocal Outlier Factor Results:")
print(f"    - Normal samples (value=1): {lof_normal}")
print(f"    - Anomalies (value=-1): {lof_anomalies}")
print(f"    - LOF score range: [{lof_scores.min():.4f}, {lof_scores.max():.4f}]")
print(f"    - Mean LOF score (normal): {lof_scores[lof_pred == 1].mean():.4f}")
print(f"    - Mean LOF score (anomaly): {lof_scores[lof_pred == -1].mean():.4f}")

joblib.dump(lof, os.path.join(output_dir, 'lof_model.pkl'))
print("\n    ✓ Saved: lof_model.pkl")

# ============================================================================
# COMPARISON & VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON & VISUALIZATION")
print("=" * 80)

# Create comparison figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Unsupervised Anomaly Detection Methods Comparison', fontsize=16, fontweight='bold')

# Plot 1: Isolation Forest scores
axes[0, 0].hist(iso_scores[iso_pred == 1], bins=50, alpha=0.7, label='Normal', color='blue')
axes[0, 0].hist(iso_scores[iso_pred == -1], bins=50, alpha=0.7, label='Anomaly', color='red')
axes[0, 0].set_title('Isolation Forest Anomaly Scores')
axes[0, 0].set_xlabel('Anomaly Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Autoencoder reconstruction error
axes[0, 1].hist(test_mae[ae_pred == 1], bins=50, alpha=0.7, label='Normal', color='blue')
axes[0, 1].hist(test_mae[ae_pred == -1], bins=50, alpha=0.7, label='Anomaly', color='red')
axes[0, 1].axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
axes[0, 1].set_title('Autoencoder Reconstruction Error (MAE)')
axes[0, 1].set_xlabel('MAE')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: LOF scores
axes[1, 0].hist(lof_scores[lof_pred == 1], bins=50, alpha=0.7, label='Normal', color='blue')
axes[1, 0].hist(lof_scores[lof_pred == -1], bins=50, alpha=0.7, label='Anomaly', color='red')
axes[1, 0].set_title('Local Outlier Factor (LOF) Scores')
axes[1, 0].set_xlabel('LOF Score')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Anomaly count comparison
methods = ['Isolation Forest', 'Autoencoder', 'LOF']
anomaly_counts = [iso_anomalies, ae_anomalies, lof_anomalies]
normal_counts = [iso_normal, ae_normal, lof_normal]

x = np.arange(len(methods))
width = 0.35
axes[1, 1].bar(x - width/2, normal_counts, width, label='Normal', color='skyblue')
axes[1, 1].bar(x + width/2, anomaly_counts, width, label='Anomalies', color='lightcoral')
axes[1, 1].set_title('Anomaly Detection Count Comparison')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(methods)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'unsupervised_comparison.png'), dpi=300, bbox_inches='tight')
print("\n    ✓ Saved: unsupervised_comparison.png")
plt.close()

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Method': ['Isolation Forest', 'Autoencoder', 'LOF'],
    'Anomalies Detected': [iso_anomalies, ae_anomalies, lof_anomalies],
    'Normal Samples': [iso_normal, ae_normal, lof_normal],
    'Anomaly %': [
        f"{(iso_anomalies / len(X_test)) * 100:.2f}%",
        f"{(ae_anomalies / len(X_test)) * 100:.2f}%",
        f"{(lof_anomalies / len(X_test)) * 100:.2f}%"
    ]
})

comparison_df.to_csv(os.path.join(output_dir, 'unsupervised_comparison.csv'), index=False)
print("\n    ✓ Saved: unsupervised_comparison.csv")
print("\n" + comparison_df.to_string(index=False))

# Summary statistics
summary = {
    'isolation_forest': {
        'anomalies': int(iso_anomalies),
        'normal': int(iso_normal),
        'anomaly_percentage': float((iso_anomalies / len(X_test)) * 100),
        'mean_score': float(iso_scores.mean()),
        'anomaly_mean_score': float(iso_scores[iso_pred == -1].mean())
    },
    'autoencoder': {
        'anomalies': int(ae_anomalies),
        'normal': int(ae_normal),
        'anomaly_percentage': float((ae_anomalies / len(X_test)) * 100),
        'threshold': float(threshold),
        'mean_mae': float(test_mae.mean()),
        'anomaly_mean_mae': float(test_mae[ae_pred == -1].mean())
    },
    'lof': {
        'anomalies': int(lof_anomalies),
        'normal': int(lof_normal),
        'anomaly_percentage': float((lof_anomalies / len(X_test)) * 100),
        'mean_score': float(lof_scores.mean()),
        'anomaly_mean_score': float(lof_scores[lof_pred == -1].mean())
    }
}

with open(os.path.join(output_dir, 'unsupervised_summary.json'), 'w') as f:
    json.dump(summary, f, indent=4)
print("    ✓ Saved: unsupervised_summary.json")

print("\n" + "=" * 80)
print("UNSUPERVISED ANOMALY DETECTION COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nOutput Directory: {output_dir}")
print("\nGenerated Files:")
print(f"  - isolation_forest.pkl")
print(f"  - autoencoder_model.h5")
print(f"  - lof_model.pkl")
print(f"  - unsupervised_comparison.png")
print(f"  - unsupervised_comparison.csv")
print(f"  - unsupervised_summary.json")