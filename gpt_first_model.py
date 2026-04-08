import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "/Users/kuberapaul/Desktop/url detection 2/final_dataset_with_all_features_v3.1.csv"
OUTPUT_DIR = "/Users/kuberapaul/Desktop/url detection 2/models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GPT FIRST BASELINE - DECISION TREE (SUPERVISED)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print("\n[1] Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"    - Shape: {df.shape}")
print(f"    - Columns: {df.columns.tolist()}")

# Identify target column (same logic as train_model.py)
possible_targets = [
    "label",
    "target",
    "class",
    "Label",
    "Target",
    "Class",
    "is_malicious",
    "malicious",
]
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    target_col = df.columns[-1]

print(f"    - Target column: {target_col}")

# ---------------------------------------------------------------------------
# 2. Basic preprocessing (first attempt)
# ---------------------------------------------------------------------------
print("\n[2] Basic preprocessing (first attempt)...")

X = df.drop(columns=[target_col])
y = df[target_col]

print(f"    - Features shape before cleaning: {X.shape}")
print(f"    - Target distribution:\n{y.value_counts()}")

# Drop obviously high-cardinality identifier columns (no labels needed here)
cols_to_drop = ["url", "type", "domain", "scan_date"]
cols_to_drop = [c for c in cols_to_drop if c in X.columns]
if cols_to_drop:
    print(f"    - Dropping high-cardinality columns: {cols_to_drop}")
    X = X.drop(columns=cols_to_drop)

# Handle categorical features (very simple first-attempt strategy)
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    print(f"    - Categorical columns found: {cat_cols}")
    for c in cat_cols:
        X[c] = X[c].fillna("missing")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print(f"    - Shape after one-hot encoding: {X.shape}")

# Fill missing numeric values with mean
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if X[c].isnull().sum() > 0:
        X[c] = X[c].fillna(X[c].mean())

print(f"    - Missing values after cleaning: {X.isnull().sum().sum()}")
print(f"    - Final feature shape: {X.shape}")

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
print("\n[3] Train / test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"    - X_train: {X_train.shape}")
print(f"    - X_test:  {X_test.shape}")
print(f"    - y_train distribution:\n{y_train.value_counts()}")

# ---------------------------------------------------------------------------
# 4. Train simple Decision Tree (first attempt)
# ---------------------------------------------------------------------------
print("\n[4] Training Decision Tree (very simple, underfitting first attempt)...")

# Intentionally shallow / conservative tree to behave like a naive first attempt
# This will typically underfit compared to a well-tuned model in train_model.py
dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=42,
)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
print("\n[5] Evaluation of GPT first Decision Tree model...")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"    Accuracy:  {accuracy:.4f}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall:    {recall:.4f}")
print(f"    F1-Score:  {f1:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# 6. Simple confusion matrix figure
# ---------------------------------------------------------------------------
print("\n[6] Saving confusion matrix figure...")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("GPT First Decision Tree - Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "gpt_first_decision_tree_confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"    - Saved: {cm_path}")

# ---------------------------------------------------------------------------
# 7. Metrics bar chart (GPT accuracy fig 1)
# ---------------------------------------------------------------------------
print("\n[7] Saving GPT metrics figure (Fig 1)...")

metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(6, 5))
bars = plt.bar(metrics_names, metrics_values, color=["skyblue", "lightgreen", "salmon", "plum"])
plt.ylim(0, 1.0)

for bar, v in zip(bars, metrics_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.01,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.ylabel("Score")
plt.title("GPT Decision Tree Metrics - Fig 1")
plt.tight_layout()
acc_path = os.path.join(OUTPUT_DIR, "gpt_accuracy_fig_1.png")
plt.savefig(acc_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"    - Saved: {acc_path}")

print("\n" + "=" * 80)
print("GPT FIRST BASELINE DECISION TREE COMPLETED")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
