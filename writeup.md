# Vitto Credit Default Risk Assessment
### *Executive Technical Write-up*
**Author:** Biprayan Choudhuri  
**Date:** May 26, 2026

---

## 1. Approach

Our overarching strategy was to shift the credit risk assessment away from static demographic indicators and focus entirely on rolling behavioral trends. We accomplished this through a rigorous end-to-end data science lifecycle:

* **Executive Strategy & Findings:** Our core finding fundamentally shifts how credit risk should be assessed: behavioral patterns are exponentially more predictive than demographic indicators. The single strongest delinquency signal is the cumulative number of months a customer's payments have been delayed.
* **Data Quality & Integrity:** Rather than discarding anomalous data, we resolved undocumented `EDUCATION` (0, 5, 6) and `MARRIAGE` (0) codes by mapping them to "Other". We specifically retained 3,932 negative `BILL_AMT` records, correctly interpreting them as overpayments (credit balances) rather than errors, as this is a strong indicator of low credit risk.
* **Behavioral Feature Engineering:** We engineered three custom variables representing rolling 6-month historical behaviors:
  * **`AVG_UTIL_RATE`:** Mean ratio of statement balances to credit limit (captures exposure risk).
  * **`AVG_PAY_RATIO`:** Mean payment-to-bill ratio (tracks repayment consistency).
  * **`TOTAL_DELAY_MONTHS`:** The cumulative count of payment delays over the past 6 months. This proved to be the single most powerful feature in the dataset, accounting for over 45% of the model's predictive weight.
* **Fairness & Operationalization (Bonus):** To ensure ethical lending, we audited the model for demographic disparities, noting higher False Positive Rates (FPR) for males and high school graduates. Finally, to bridge the gap between data science and frontline operations, we operationalized the model by building the interactive **Vitto Risk Portal** using Streamlit.

---

## 2. Tradeoffs

A significant tradeoff in this project revolved around handling the **22.1% target class imbalance** (defaults vs. non-defaults).

* **Cost-Sensitive Learning vs. Synthetic Sampling:** We explicitly rejected synthetic sampling techniques like SMOTE. In datasets heavily reliant on one-hot encoded categorical variables (like demographics), SMOTE generates unrealistic "synthetic" credit profiles in high-dimensional dummy space. 
* **The Decision:** Instead, we chose **cost-sensitive learning** using `class_weight="balanced"` (Logistic Regression) and `scale_pos_weight` (XGBoost).
* **Business Alignment:** This tradeoff prioritizes native data distributions while penalizing misclassified defaults more heavily. It perfectly aligns with Vitto's financial reality: a False Negative (approving a borrower who defaults) is exponentially costlier than a False Positive (denying a good borrower).

---

## 3. Model Choices

We evaluated a linear baseline against a tree-based ensemble method to balance interpretability with predictive power. 

* **The Baseline:** We started with a **Logistic Regression** model because it offers highly transparent, easily auditable coefficients.
* **The Champion Model:** **XGBoost** was selected as our final champion model. While Logistic Regression assumes linear relationships, credit risk is highly non-linear (e.g., low utilization is safe, moderate is healthy, but extreme utilization spikes risk exponentially). XGBoost natively captures these complex, non-linear interactions.
* **Solving the Interpretability Tradeoff:** We resolved the "black box" nature of XGBoost by integrating **SHAP (SHapley Additive exPlanations)**, which provides full local and global interpretability for every individual prediction to satisfy regulatory auditing requirements.

**Model Performance & Efficacy**
To guarantee stability, the XGBoost model was validated using stratified 5-fold cross-validation. At our chosen threshold, the model successfully flags **57% of actual defaults** before they happen, while maintaining strong overall discrimination.

| Metric | Logistic Regression (Baseline) | XGBoost (Champion Model) |
| :--- | :---: | :---: |
| **Precision (Class 1)** | 0.4600 | **0.4500** |
| **Recall (Class 1)** | 0.5700 | **0.5700** |
| **F1-Score (Class 1)** | 0.5100 | **0.5000** |
| **Test AUC-ROC** | 0.7500 | **0.7536** |
| **5-Fold CV AUC-ROC** | *N/A* | **0.7581 ± 0.0073** |

---

## 4. What I'd Improve Given More Time

Given an additional 2–3 working days, I would implement the following enhancements to mature this decisioning asset for production deployment:

1. **Temporal Validation (Out-Of-Time Split):**  
   Currently, we use a random train/test split. In production lending, we must use an out-of-time (OOT) split (e.g., training on April–August and testing on September). This rigorously validates the model's ability to generalize against macroeconomic shifts over time.
2. **Threshold-Moving & Expected Value Framework:**  
   Instead of relying purely on `scale_pos_weight`, I would output raw probabilities and perform threshold optimization against a specific financial loss matrix (e.g., "A false positive costs us $500 in lost revenue; a false negative costs us $5,000 in charge-offs") to maximize net portfolio profit.
3. **Hyperparameter Optimization:**  
   Implement Bayesian optimization (via `Optuna` or `Hyperopt`) to fine-tune XGBoost parameters (e.g., `max_depth`, `learning_rate`, `subsample`). This could reliably squeeze an additional 2-4% of performance out of the AUC-ROC.
4. **Alternative Data Integration:**  
   Integrate macroeconomic indicators (e.g., local inflation rates, regional employment figures) to provide broader economic context to an individual borrower's utilization and repayment behaviors.