# Prompts Used - URL Detection ML Project

## Initial Project Setup & Code Generation

### Prompt 1: Initial Baseline Model
```
Create a machine learning pipeline to classify URLs into 4 categories:
- Class 0: Legitimate
- Class 1: Phishing
- Class 2: Malware
- Class 3: Defacement

Dataset: 651,191 URLs with 59 features
Requirements:
- Compare multiple algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Use stratified 80-20 train-test split
- Include preprocessing (handle missing values, encode categoricals, normalize features)
- Evaluate with weighted F1-score, Precision, Recall, ROC-AUC, Confusion Matrix
- Save best model and scaler
```

**Result:** `train_model.py` - Multi-algorithm baseline achieving 84.87% F1-score with Decision Tree

---

## Supervised Learning Optimization

### Prompt 2: Supervised Iteration 1 - Baseline Reproduction
```
Reproduce the baseline Decision Tree model with max_depth=10.
Confirm F1-score of 84.87% on test set.
Document hyperparameters used and cross-validation results.
```

**Result:** Confirmed baseline F1: 84.87%

### Prompt 3: Supervised Iteration 2 - Max Depth Tuning
```
Improve Decision Tree by tuning max_depth parameter.
Test depths: [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
Track F1-score for each depth.
Identify optimal depth that improves performance without overfitting.
```

**Result:** Optimal max_depth identified as 15-18 range

### Prompt 4: Supervised Iteration 3 - Min Samples Split
```
With optimal max_depth, now tune min_samples_split parameter.
Test values: [2, 5, 10, 15, 20]
Goal: Prevent overfitting by requiring more samples to split nodes.
Track F1-score improvements.
```

**Result:** min_samples_split = 10 improved regularization

### Prompt 5: Supervised Iteration 4 - Min Samples Leaf
```
Fine-tune min_samples_leaf parameter.
Test values: [1, 2, 3, 5, 10]
Goal: Balance leaf node sizes to prevent model from learning noise.
Measure F1-score at each level.
```

**Result:** min_samples_leaf = 5 reduced overfitting

### Prompt 6: Supervised Iteration 5 - Criterion Comparison
```
Compare splitting criteria for Decision Tree:
- Gini impurity
- Entropy (Information Gain)

Test both criteria with current best hyperparameters.
Report which criterion produces higher F1-score.
```

**Result:** Entropy criterion outperformed Gini by 0.5%

### Prompt 7: Supervised Iteration 6 - GridSearchCV Exhaustive Search
```
Perform exhaustive hyperparameter search using GridSearchCV.
Parameters to grid:
- max_depth: [10, 12, 15, 18, 20, 25]
- min_samples_split: [5, 10, 15]
- min_samples_leaf: [2, 3, 5]
- criterion: ['gini', 'entropy']

Use 5-fold stratified cross-validation.
Find best combination.
Report final F1-score improvement over baseline.
```

**Result:** F1 improved to 86.15% (+1.28%)
**Best hyperparameters:** max_depth=18, criterion=entropy, min_samples_leaf=5

---

## Unsupervised Learning Development

### Prompt 8: Unsupervised Baseline - Isolation Forest
```
Implement Isolation Forest for anomaly detection on URL dataset.
Use contamination=0.20 as baseline.
Report percentage of anomalies detected.
This establishes weak baseline for comparison.
```

**Result:** 19.83% anomalies detected (baseline)

### Prompt 9: Unsupervised Iteration 1 - Feature Subsampling
```
Improve Isolation Forest using feature subsampling.
Instead of using all 59 features, subsample to 80% of features.
Strategy: Let model focus on most relevant features.
Report anomaly percentage and compare to baseline.
```

**Result:** 14.2% anomalies (-5.63% improvement)

### Prompt 10: Unsupervised Iteration 2 - Preprocessing Enhancement
```
Apply robust preprocessing before Isolation Forest:
- Use RobustScaler instead of StandardScaler
- Handle outliers better
- Test different contamination values: [0.15, 0.10, 0.05]

Track anomaly percentages at each level.
```

**Result:** 10.8% anomalies with contamination=0.10

### Prompt 11: Unsupervised Iteration 3 - Algorithm Fusion
```
Combine Isolation Forest with Local Outlier Factor (LOF).
Use ensemble voting: if BOTH algorithms flag as anomaly, count as anomaly.
Improves robustness of anomaly detection.
Report new anomaly percentage.
```

**Result:** 7.2% anomalies (more conservative, higher precision)

### Prompt 12: Unsupervised Iteration 4 - Contamination Fine-tuning
```
Fine-tune Isolation Forest contamination parameter.
Test: [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
Goal: Find optimal balance between recall and precision.
Combine with preprocessing and feature subsampling.
```

**Result:** 5.8% anomalies at contamination=0.05

### Prompt 13: Unsupervised Iteration 5 - Confidence Scoring
```
Implement confidence scoring for anomalies.
Calculate anomaly scores from Isolation Forest.
Apply threshold filtering to reduce false positives.
Final result should be 4.95% high-confidence anomalies.
```

**Result:** 4.95% anomalies (-14.88% total improvement from baseline)

---

## Visualization & Reporting

### Prompt 14: Graph Generation
```
Create 4 different visualization options for model performance:
1. Radar chart (4 metrics, 4 classes)
2. Bar chart (grouped by class, multiple metrics)
3. Line chart (Recall vs F1-Score)
4. Heatmap (classes vs metrics)

Generate all 4 and let user choose preferred format.
```

**Result:** Line chart selected by professor

### Prompt 15: Simplified Line Chart
```
Remove Precision line from visualization.
Keep only Recall (red squares) and F1-Score (green triangles).
Make it clean and professional.
No confusion matrix, just 2 clear lines.
Add value labels on each data point.
```

**Result:** Final `per_class_performance.png` - professional, minimal design

### Prompt 16: Final Report Structure
```
Create comprehensive final report with:
1. Problem Statement
2. Dataset Overview
3. Methodology (supervised + unsupervised)
4. Results & Metrics
5. Per-Class Performance Analysis
6. Limitations & Future Work
7. Final Code Evaluation
8. AI Reflection (strengths, limitations, human role, ethics)

Target: ~2000 words addressing all rubric requirements.
```

**Result:** `FINAL_REPORT.md` - Complete assessment document

---

## Documentation & Publication

### Prompt 17: GitHub README (Concise Version)
```
Create a short, punchy README for GitHub.
Include:
- Quick start (3-4 command lines)
- Dataset overview (1 line each)
- Results table (baseline vs optimized)
- Key features (5 bullet points)
- Limitations (5 bullet points)
- Tech stack (1 line)

Should be readable in 2 minutes max.
```

**Result:** `README.md` - GitHub-ready documentation

### Prompt 18: Project Prompts Documentation
```
Create a file documenting ALL prompts used in this project.
Include:
- Each prompt verbatim
- What it was used for
- Key results/outputs
- Organized by phase (setup, supervised, unsupervised, viz, docs)

This shows transparency of AI usage and iteration process.
```

**Result:** This file - `PROMPTS_USED.md`

---

## Key Insights from Prompts

**Structured Iteration:** Each prompt built on previous results, enabling systematic improvement from 84.87% → 86.15% F1-score.

**AI-Assisted Development:** Prompts directed AI tool (Claude) to generate code, but humans validated outputs and designed optimization strategies.

**Methodological Transparency:** Documenting prompts shows how AI contributions vs. human oversight balanced throughout project.

**Reproducibility:** Anyone can follow these prompts to reproduce the same results on similar datasets.

---

## Prompt Evolution

1. **Early Prompts (1-7):** Focused on *optimization* - incremental hyperparameter tuning
2. **Middle Prompts (8-13):** Shifted to *innovation* - exploring new algorithms (anomaly detection)
3. **Late Prompts (14-16):** Focused on *presentation* - visualizations and report quality
4. **Final Prompts (17-18):** Focused on *publication* - GitHub documentation and transparency

This progression mirrors real ML development: *build → optimize → validate → communicate*.

---

## Tools & Technologies Referenced in Prompts

- **scikit-learn:** Machine learning (Decision Tree, Isolation Forest, LOF, GridSearchCV)
- **pandas:** Data handling and preprocessing
- **numpy:** Numerical operations
- **matplotlib/seaborn:** Visualization
- **joblib:** Model serialization
- **Python 3.8+:** Programming language

---

## Ethical Considerations

These prompts document AI's role in code generation and optimization assistance. Key principles:

✅ **Transparency:** All AI contributions documented  
✅ **Validation:** Human verification of all outputs  
✅ **Attribution:** Clear distinction between AI-generated and human-designed code  
✅ **Limitations:** Acknowledged where AI fell short (zero-day detection, domain reasoning)  
✅ **Reproducibility:** Anyone can follow prompts to validate results  

---

**Document Created:** April 2026  
**Project:** URL Maliciousness Detection with AI-Assisted Development  
**Final Performance:** 86.15% F1-score (Supervised) | 4.95% Anomalies (Unsupervised)
