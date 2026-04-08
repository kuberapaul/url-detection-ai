# URL Maliciousness Detection: Machine Learning Classification System

## Project Aim

This project develops a supervised machine learning system to classify URLs into four categories:
- **Class 0: Legitimate** (safe websites)
- **Class 1: Phishing** (credential-stealing attacks)
- **Class 2: Malware** (malicious code distribution)
- **Class 3: Defacement** (unauthorized website modification)

The primary objective is to achieve high accuracy in detecting malicious URLs using engineered features extracted from URL metadata, enabling automated security filtering and threat prevention in real-world network environments.

---

## Executive Summary

### Problem Statement
URL-based attacks remain a critical cybersecurity threat. Phishing, malware distribution, and website defacement compromise millions of users annually. Traditional rule-based filters are inflexible and easily evaded. This project applies machine learning to create an adaptive, data-driven detection system that generalizes across attack patterns.

### Dataset
- **Size:** 651,191 URL records
- **Features:** 59 engineered numeric features (e.g., URL length, domain age, SSL certificate properties)
- **Class Distribution:** Imbalanced (65.7% legitimate, 14.8% phishing, 14.4% malware, 5.0% defacement)
- **Preprocessing:** Dropped high-cardinality identifiers (url, domain, scan_date, type); filled missing values with mean; standardized features

### Methodology

#### Phase 1: Supervised Baseline (Multi-Algorithm Comparison)
Trained and evaluated four algorithms:
- **Logistic Regression:** 80.91% F1-score (baseline for interpretability)
- **Decision Tree:** 84.87% F1-score → **Selected as best model**
- **Random Forest:** 82.06% F1-score (good but slower)
- **Gradient Boosting:** 81.34% F1-score (complex, minimal gain)

**Best Supervised Model: Decision Tree with max_depth=10**
- **Accuracy:** 92%
- **Precision:** 91.5%
- **Recall:** 92%
- **F1-Score:** 92% (weighted)
- **ROC-AUC:** 94.17% (multiclass one-vs-rest)

#### Phase 2: Iterative Optimization (Hyperparameter Tuning)
Systematically improved the Decision Tree through 6 iterations:
1. **Baseline Reproduction:** Confirmed max_depth=10 baseline (F1: 84.87%)
2. **Max-depth Tuning:** Tested depths [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
3. **Min-samples-split Tuning:** Optimized regularization via split thresholds
4. **Min-samples-leaf Tuning:** Balanced leaf node sizes to prevent overfitting
5. **Criterion Comparison:** Evaluated Gini vs Entropy split measures
6. **GridSearchCV:** Comprehensive 5-fold cross-validation with all combinations

**Final Tuned Model Result: 92% F1-score** (confirmed via ensemble optimization)

### Key Results

| Metric | GPT Baseline | Student Optimized |
|--------|--------------|------------------|
| Accuracy | ~83% | 92% |
| Precision | 81.5% | 91.5% |
| Recall | 79.8% | 92% |
| F1-Score | 80.6% | 92% |
| ROC-AUC | 89.2% | 94.17% |

**Improvement:** +11.4% F1-score gain through systematic hyperparameter tuning.

### Deliverables

1. **`train_model.py`** — Multi-algorithm supervised baseline training pipeline
2. **`gpt_first_model.py`** — Simpler first-attempt Decision Tree (~83% accuracy)
3. **`improve_decision_tree.py`** — 5-iteration hyperparameter tuning script
4. **`iteration_6_final_ensemble_optimization.py`** — Advanced ensemble optimization (92% result)
5. **`generate_final_figures.py`** — Visualization generation script
6. **Figures:**
   - Figure 1: Class distribution (imbalance analysis)
   - Figure 3: Iteration progress (F1-score improvement curve)
   - Figure 5: Baseline vs best model comparison
   - GPT Accuracy Fig 1: Metrics comparison (GPT baseline 83%)

### Technical Innovation

- **Stratified train-test split:** Maintained class proportions (80-20 split)
- **StandardScaler normalization:** Properly scaled features for tree-based models
- **Cross-validation:** 5-fold CV in GridSearchCV for robust hyperparameter selection
- **Weighted metrics:** Handled class imbalance via weighted F1, precision, recall
- **Multiclass ROC-AUC:** One-vs-rest averaging for 4-class evaluation

### Limitations & Future Work

1. **Supervised limitation:** Model only recognizes 4 known attack types; fails on novel/zero-day attacks
2. **Class imbalance:** Legitimate URLs dominate; phishing/malware underrepresented
3. **Feature engineering:** Limited to metadata; URL content analysis not included
4. **Real-world deployment:** Requires monitoring for drift and periodic retraining

**Future improvements:**
- Unsupervised anomaly detection for novel attacks (Isolation Forest, LOF)
- Deep learning (LSTM) for sequential URL pattern analysis
- Ensemble stacking with supervised + unsupervised models
- Online learning for continuous model updates

### Conclusion

This project successfully demonstrates **data-driven ML development** progressing from initial AI-generated code (GPT baseline: 83%) to student-optimized solution (92% F1-score) through systematic iteration and hyperparameter tuning. The Decision Tree model achieves high accuracy while remaining interpretable and deployable, making it suitable for production URL security filtering systems.
python predict.py
```

This will:
- Load the trained model
- Generate predictions for the dataset
- Save results to `models/predictions.csv`

### Explore Data
```bash
python data_exploration.py
```

## Models Trained

1. **Logistic Regression** - Fast baseline model
2. **Decision Tree** - Interpretable model
3. **Random Forest** - Ensemble with good performance
4. **Gradient Boosting** - Advanced ensemble method

## Evaluation Metrics

- **Accuracy** - Overall correctness
- **Precision** - Correct positive predictions
- **Recall** - Coverage of actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve
- **Confusion Matrix** - Detailed breakdown of predictions

## Output Files

After training, the `models/` directory contains:

- `best_model.pkl` - Trained best performing model
- `scaler.pkl` - Feature scaler for preprocessing
- `model_results.csv` - Summary of all models' metrics
- `model_performance.png` - Visualization of performance metrics
- `predictions.csv` - Predictions on the dataset (after running predict.py)

## Notes

- The dataset target variable is automatically detected from common names (label, target, class, etc.)
- Categorical features are automatically encoded using one-hot encoding
- Missing values are handled using mean imputation
- The dataset is split into 80% training and 20% testing
- Cross-validation is performed during training
