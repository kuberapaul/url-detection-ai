# URL Maliciousness Detection

Classifies URLs into four categories: **Legitimate**, **Phishing**, **Malware**, **Defacement**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py


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

## Citations

Software Libraries & Frameworks:

Pedregosa, Fabian, et al. "Scikit-Learn: Machine Learning in Python." Journal of Machine Learning Research, vol. 12, 2011, pp. 2825-2830.

Harris, Charles R., et al. "Array Programming with NumPy." Nature, vol. 585, no. 7825, 2020, pp. 357-362, https://doi.org/10.1038/s41586-020-2649-2.

McKinney, Wes. "Data Structures for Statistical Computing in Python." Proceedings of the 9th Python in Science Conference, 2010, pp. 56-61.

Hunter, John D. "Matplotlib: A 2D Graphics Environment." Computing in Science & Engineering, vol. 9, no. 3, 2007, pp. 90-95, https://doi.org/10.1109/MCSE.2007.55.

Waskom, Michael L. "Seaborn: Statistical Data Visualization." Journal of Open Source Software, vol. 6, no. 60, 2021, p. 3021, https://doi.org/10.21105/joss.03021.


Machine Learning Methodologies:

Breiman, Leo. "Classification and Regression Trees." Chapman and Hall, 1984.

Breiman, Leo. "Random Forests." Machine Learning, vol. 45, no. 1, 2001, pp. 5-32, https://doi.org/10.1023/A:1010933404324.

Friedman, Jerome H. "Greedy Function Approximation: A Gradient Boosting Machine." Annals of Statistics, vol. 29, no. 5, 2001, pp. 1189-1232, https://doi.org/10.1214/aos/1013203451.

Fawcett, Tom. "An Introduction to ROC Analysis." Pattern Recognition Letters, vol. 27, no. 8, 2006, pp. 861-874, https://doi.org/10.1016/j.patrec.2005.10.010.





