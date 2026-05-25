# 💳 Vitto Credit Default Risk Analysis

This repository contains the complete analytical pipeline, predictive model, and executive narrative designed for **Vitto**'s credit risk team. This project aims to identify the primary behavioral signals driving defaults and build a robust machine learning classifier.

---

## 1. Setup Guide

This project is built using Python 3.12+ and uses the high-performance **`uv`** package manager for fast, reliable virtual environments and dependency management.

### Prerequisites
Make sure you have Python 3.12+ and `uv` installed. If you do not have `uv`, you can install it via:
```bash
pip install uv
```

### Installation & Execution Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/CodeNeuron58/vitto-credit-risk-assessment
   cd vitto-credit-risk-assessment
   ```

2. **Initialize Environment & Install Dependencies:**
   Run the following command to automatically create a `.venv` virtual environment and sync all dependencies from `pyproject.toml` and `uv.lock`:
   ```bash
   uv sync
   ```

3. **Launch the Jupyter Notebook:**
   Start the Jupyter server in your environment to explore `Credit_Risk_Analysis.ipynb`:
   ```bash
   uv run jupyter notebook Credit_Risk_Analysis.ipynb
   ```

4. **Launch the Vitto Risk Portal (Streamlit App):**
   We have also provided a Streamlit dashboard for real-time risk assessment.
   ```bash
   uv run streamlit run app/app.py
   ```

---

## 2. Dataset Source

The data utilized in this project is based on the default behavior of credit card clients in Taiwan. 

* **Dataset:** Default of Credit Card Clients Dataset (30,000 records, 25 variables, Apr–Sep 2005)
* **Source:** UCI Machine Learning Repository - [Link to Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* **Citation:** Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2471-2475.

---

## 3. Methodology Summary

Our analytical approach rigorously followed an end-to-end data science lifecycle to ensure robustness and business utility:

1. **Exploratory Data Analysis & Cleaning:** 
   * Loaded the dataset and identified anomalies (e.g., negative `BILL_AMT` indicating overpayment, undocumented categorical values).
   * Mapped undocumented values in `EDUCATION` (0, 5, 6) and `MARRIAGE` (0) to "Other" categories to preserve data integrity.
2. **Feature Engineering:**
   * Shifted focus from static demographic data to rolling 6-month behavioral trends, creating composite metrics for utilization, payment ratios, and delinquency.
3. **Data Preparation:**
   * Addressed the 22.1% default class imbalance natively using cost-sensitive learning (`class_weight="balanced"` and `scale_pos_weight`) rather than synthetic sampling like SMOTE, which distorts high-dimensional dummy variables.
   * Split data into 80/20 train-test sets using **stratified sampling**, standardizing features securely on the training set to prevent data leakage.
4. **Model Training & Evaluation:**
   * Trained a baseline Logistic Regression model against an XGBoost classifier. 
   * XGBoost was selected as the champion model to handle non-linear risk interactions, validated thoroughly via 5-Fold Cross-Validation (CV AUC-ROC: **0.7572 ± 0.0060**).
5. **Auditing & Explainability (Bonuses):**
   * Conducted SQL-based demographic analysis.
   * Audited the model for algorithmic fairness, noting disparities in False Positive Rates (FPR) across Gender and Education.
   * Deployed SHAP (SHapley Additive exPlanations) to provide local and global interpretability for the XGBoost "black box" decisions.

---

## 4. Feature List

In addition to standard demographic variables (encoded) and the provided 6-month billing history, we engineered three custom behavioral features designed to capture borrower risk dynamics:

| Feature Name | Type | Formula / Logic | Business Rationale |
| :--- | :--- | :--- | :--- |
| **`AVG_UTIL_RATE`** | Numeric | `mean(BILL_AMT_x / LIMIT_BAL)` | Measures average credit exposure. High utilization paired with low repayments indicates liquidity distress. |
| **`AVG_PAY_RATIO`** | Numeric | `mean(PAY_AMT_x / BILL_AMT_x)` (where `BILL_AMT_x > 0`) | Measures repayment consistency. Tracks what fraction of the outstanding bill the borrower actually pays off. |
| **`TOTAL_DELAY_MONTHS`**| Numeric | `count(PAY_x > 0)` | Delinquency duration. The total number of months in the past 6 months where the payment was 1+ months overdue. |

---

## 5. Key Findings

Our core analytical finding is that **behavior trumps demographics.** A customer's repayment history and utilization rate over the last six months are exponentially more predictive of default than their gender, age, or education level.

* **Primary Signal (`TOTAL_DELAY_MONTHS`):** The cumulative number of months a borrower is late is the single most powerful warning signal, accounting for over **45% of the total predictive weight** in our model. Accumulating three or more late months is a near-certain warning signal of impending default.
* **Secondary Signal (`PAY_0`):** A borrower's payment status in the most recent month is the next strongest indicator. Customers currently 2+ months in arrears default at a rate of over **69%**.
* **Model Efficacy:** At our operating threshold, the XGBoost model successfully flags **58% of actual defaults** (Recall) before they happen. It ranks a high-risk borrower above a low-risk borrower roughly 3 out of 4 times (AUC-ROC: 0.752), providing a solid baseline for an automated early-intervention system.

### Recommended Actions for the Credit Team
1. **Early-Warning Delinquency Trigger:** Implement a proactive outreach rule flagging any cardholder who records **2 or more delayed payment months** in a rolling 6-month window for soft interventions (e.g., SMS alerts).
2. **Behavioral Risk Limit Cap:** Replace static annual reviews with dynamic monthly reviews. Automatically cap the credit limit for any active borrower whose payment ratio drops below 10% while utilization climbs past 80%.