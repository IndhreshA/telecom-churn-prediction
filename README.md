# Telecom Customer Churn Prediction

Predicting which telecom customers are likely to churn, with a focus on evaluating models correctly on imbalanced data.

**Course:** Applied AI/ML, The University of Texas at Dallas (Nov–Dec 2025)
**Tools:** Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

## Problem
About 27% of customers in the dataset churned. With an imbalance like this, a model can reach high accuracy just by predicting "no churn" for everyone, while catching zero churners.

## Approach
1. **EDA:** explored feature distributions and their relationships with churn.
2. **Preprocessing:** scaled numeric features and one-hot encoded categorical features inside a Scikit-learn `Pipeline` / `ColumnTransformer`.
3. **Models:** K-Nearest Neighbors, Decision Tree, Random Forest, and Logistic Regression.
4. **Tuning:** `GridSearchCV` with 5-fold cross-validation.
5. **Evaluation:** classification reports (precision, recall, F1) and confusion matrices, not just accuracy.

## Key finding
- The initial models reached about 73% accuracy, which matched the majority-class baseline. They predicted almost no churners (churn-class F1 of 0.00–0.02).
- Re-tuning for F1 and adding class weighting (`class_weight="balanced"`) raised churn recall from **0% to 47%** with a balanced Random Forest (churn-class F1 of about 0.33).
- Precision stayed low overall, which suggests the available features carry limited churn signal. Better features (e.g. usage trends, support history) would be the next step.

## Takeaway
Accuracy alone can be misleading on imbalanced data. Recall, precision, and F1 for the minority class give a truer picture of how useful a churn model is.

## Files
- `Telecom Churn.ipynb`: full analysis, modeling, and evaluation
- `TelecomCo_ChurnData.csv`: dataset

## How to run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter lab
```
Open the notebook, make sure the CSV path in the data-loading cell points to the dataset, and run all cells.
