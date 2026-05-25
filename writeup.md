# Vitto Credit Default Risk Assessment
### *Executive Technical Write-up for the Credit Risk Committee*
**Author:** Biprayan Choudhuri  
**Date:** May 26, 2026

---

## 1. Executive Summary & Business Value

This document details the end-to-end development of an advanced credit risk decisioning pipeline designed for **Vitto's digital lending operations**. Utilizing behavioral and demographic data from 30,000 credit card clients, we engineered a machine learning classification engine that identifies borrowers at high risk of default *before* severe delinquency occurs. 

Our core finding fundamentally shifts how credit risk should be assessed: **behavioral patterns are exponentially more predictive than static demographic indicators**. 

The single strongest delinquency signal is the cumulative number of months a customer's payments have been delayed over the past six months (`TOTAL_DELAY_MONTHS`). By deploying our champion XGBoost classifier, Vitto can proactively identify **57% of impending defaults** (Recall) while maintaining a high overall discrimination accuracy (AUC-ROC: **0.754**, 5-fold CV AUC-ROC: **0.758**). 

### Key Business Actions
1. **Early-Warning Delinquency Trigger:** Implement automated, soft-touch interventions (e.g., SMS reminders, flexible repayment offers) for any cardholder who records 2 or more delayed payment months in a rolling 6-month window.
2. **Dynamic Risk-Based Limit Caps:** Transition from static annual reviews to dynamic monthly scores. Automatically cap credit limits for borrowers showing shrinking repayment-to-bill ratios combined with escalating credit utilization.

---

## 2. Methodology & Analytical Approach

Our analytical pipeline is divided into five rigorous stages:

1. **Ingestion & Data Quality Audit:**  
   Loaded the 30,000-record dataset and screened for anomalies. Documented and resolved two key data discrepancies:
   * **Education & Marriage:** Undocumented education codes (`0`, `5`, `6`) and marriage codes (`0`) were mapped into their respective "Other" categories to ensure data integrity without discarding valuable rows.
   * **Negative Balances (`BILL_AMT`):** Detected 3,932 negative billing records (indicating overpayments or credit balances) and retained them to accurately calculate credit utilization rates, as overpayment is a strong indicator of low credit risk.

2. **Behavioral Feature Engineering:**  
   To move beyond static demographic metrics, we engineered three custom variables representing rolling 6-month historical behaviors:
   * **`AVG_UTIL_RATE`:** Mean ratio of statement balances to credit limit. Captures exposure risk and liquidity distress.
   * **`AVG_PAY_RATIO`:** Mean payment-to-bill ratio. Tracks repayment consistency.
   * **`TOTAL_DELAY_MONTHS`:** The cumulative count of payment delays (where repayment status > 0) in the past 6 months. This proved to be the most powerful feature in the dataset.

3. **Data Preparation & Encoding:**  
   Categorical variables were one-hot encoded using `drop_first=True` to avoid multicollinearity (the dummy variable trap). To prevent data leakage, continuous features were standardized using a scaler fitted strictly on the training set.

4. **Model Development & Stratification:**  
   Split the data into 80% Train and 20% Test using **stratified sampling** to strictly maintain the 22.1% default class ratio across both sets. Evaluated a Logistic Regression baseline against an XGBoost classifier.

5. **Decisioning, Auditing, & Explainability:**  
   Audited model results for algorithmic fairness across demographics and extracted local and global feature importance scores via SHAP (SHapley Additive exPlanations) to ensure compliance with financial regulations.

---

## 3. Key Technical Tradeoffs & Design Decisions

### Tradeoff A: Class Imbalance Strategy (Cost-Sensitive Learning vs. Synthetic Sampling)
* **The Problem:** The target variable exhibits a 22.1% default rate. Standard machine learning classifiers optimizing for pure accuracy would default to a "majority-class classifier" (predicting no default for everyone, yielding ~78% accuracy but 0% recall of defaults).
* **The Decision:** We explicitly rejected synthetic sampling techniques like SMOTE. SMOTE can generate unrealistic "synthetic" credit profiles in high-dimensional dummy space (e.g., creating a profile that is 0.4 male and 0.6 female). Instead, we chose **cost-sensitive learning**. We set `class_weight="balanced"` in Logistic Regression and `scale_pos_weight = count(non-defaulters) / count(defaulters)` in XGBoost.
* **The Rationale:** This approach penalizes misclassified defaults more heavily without distorting the underlying data distribution, perfectly aligning the model with Vitto's business goals (where a False Negative—approving a borrower who defaults—is financially far costlier than a False Positive).

### Tradeoff B: Model Selection (Interpretability vs. Predictive Power)
* **The Comparison:** We evaluated Logistic Regression against a tree-based gradient boosted framework (XGBoost).
* **The Decision:** XGBoost was chosen as our champion model.
* **The Rationale:** While Logistic Regression offers highly transparent coefficients, it strictly assumes linear relationships. Credit risk, however, is highly non-linear (e.g., low utilization is low risk, moderate utilization is healthy, but extreme utilization spikes risk exponentially). XGBoost excels at capturing these complex, non-linear interactions. We resolved the inherent "black box" interpretability trade-off of XGBoost by implementing **SHAP values**, providing complete, auditable explanations for every individual prediction.

---

## 4. Model Performance & Validation

Both models were fitted on standard preprocessed data and evaluated on the holdout test set (6,000 cases). To ensure absolute model stability and guard against overfitting, the champion XGBoost model was validated using stratified 5-fold cross-validation.

### Performance Comparison Table

| Metric | Logistic Regression (Baseline) | XGBoost (Champion Model) | Business Rationale |
| :--- | :---: | :---: | :--- |
| **Precision (Class 1)** | 0.4600 | **0.4500** | Out of all flagged borrowers, 45% will actually default. |
| **Recall (Class 1)** | 0.5700 | **0.5700** | Vitto will proactively catch **57% of total defaults** before they happen. |
| **F1-Score (Class 1)** | 0.5100 | **0.5000** | Balanced harmonic mean of precision and recall. |
| **Test AUC-ROC** | 0.7500 | **0.7536** | Measures overall discriminative capability. |
| **5-Fold CV AUC-ROC** | *N/A* | **0.7581 ± 0.0073** | Confirms exceptional stability across folds (very low variance). |

---

## 5. Algorithmic Fairness & Ethical Audit

In modern FinTech, predictive power must be balanced with ethical lending practices. We conducted a demographic fairness audit to identify disparities in **False Positive Rates (FPR)**. A higher FPR means non-defaulting borrowers in a specific group are incorrectly flagged as high-risk, which could lead to discriminatory credit denials.

* **Gender Disparity:** Males have an FPR of **22.4%** compared to **16.9%** for females (a **5.5 percentage point disparity**). This indicates the model is slightly more conservative when evaluating male borrowers.
* **Education Disparity:** High school graduates experience a much higher FPR (**23.1%**) than graduate school alumni (**15.6%**). 
* **Strategic Recommendation:** To prevent systemic credit exclusion, Vitto should avoid using a single global risk threshold. We recommend implementing group-specific decision thresholds to align FPRs, ensuring equitable lending practices while maintaining portfolio safety.

---

## 6. Operationalization: The Vitto Risk Portal (Streamlit)

To demonstrate how this model translates from a Jupyter Notebook to a practical business tool, we developed the **Vitto Credit Risk Portal** using Streamlit. 

This interactive dashboard allows loan officers and risk managers to:
* Input a customer's demographic and financial details.
* Instantly generate a probability of default and a definitive risk classification (Low, Medium, High).
* View a dynamic gauge chart visualizing the customer's risk profile against Vitto's thresholds.
* This bridges the gap between complex data science and actionable frontline operations.

---

## 7. Future Enhancements (Given More Time)

Given an additional 2–3 working days, we would implement the following enhancements to mature the decisioning asset for full production deployment:

1. **Temporal Validation:**  
   Currently, we use a random train/test split. In production lending, we must use an out-of-time (OOT) split (e.g., training on April–August and testing on September). This rigorously validates the model's ability to handle macroeconomic shifts over time.
2. **Hyperparameter Optimization:**  
   Implement Bayesian optimization (via `Optuna` or `Hyperopt`) to fine-tune XGBoost parameters (e.g., `max_depth`, `learning_rate`, `subsample`). This could reliably squeeze an additional 2-4% of performance out of the AUC-ROC.
3. **Threshold-Moving & Expected Value Framework:**  
   Instead of using `scale_pos_weight`, we would output raw probabilities and perform threshold optimization against a specific financial loss matrix (e.g., "A false positive costs us $500 in lost revenue, a false negative costs us $5,000 in charge-offs") to maximize net portfolio profit.
4. **Alternative Data Integration:**  
   Integrate macroeconomic indicators (e.g., local inflation rates, employment figures) to provide broader context to a borrower's utilization and repayment behaviors.