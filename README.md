# URL Maliciousness Detection

Classifies URLs into four categories: **Legitimate**, **Phishing**, **Malware**, **Defacement**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Make predictions
python predict.py
```

## Dataset

- **Size:** 651,191 URLs
- **Features:** 59 engineered attributes (URL length, domain age, SSL properties, etc.)
- **Classes:** 4 (Legitimate 65.7%, Phishing 14.8%, Malware 14.4%, Defacement 5.0%)
- **Split:** 80-20 stratified train-test

## Results

| Model | F1-Score | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Baseline (Decision Tree) | 84.87% | 86.25% | 84.87% | 94.17% |
| Optimized (GridSearchCV) | 86.15% | 85.89% | 86.04% | 94.63% |
| **Improvement** | **+1.28%** | — | — | +0.46% |

**Best Model:** Decision Tree (max_depth=18, criterion=entropy, 5-fold CV)

## Optimization Approach

1. **Baseline:** Multi-algorithm comparison (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
2. **6 Iterations:** Systematic hyperparameter tuning
   - Max-depth tuning: [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
   - Min-samples-split & min-samples-leaf optimization
   - Criterion comparison: Gini vs Entropy
   - GridSearchCV: 120 configurations, 5-fold CV
3. **Final Model:** Decision Tree (max_depth=18, entropy, min_samples_leaf=5)

## Key Features

✅ Stratified train-test split (maintains class distribution)  
✅ StandardScaler normalization  
✅ 5-fold cross-validation  
✅ Weighted metrics (handles class imbalance)  
✅ Multiclass ROC-AUC (one-vs-rest averaging)

## Output Files

- `best_model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `model_results.csv` - Metrics summary
- `per_class_performance.png` - Performance visualization

## Limitations

- Fails on novel/zero-day attacks (4 known classes only)
- Class imbalance (Legitimate dominates)
- Metadata features only (no URL content analysis)
- Requires monitoring for model drift

## Tech Stack

pandas • numpy • scikit-learn • matplotlib • seaborn • joblib
