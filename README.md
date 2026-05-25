# 💳 Vitto Credit Default Risk Analysis
### *End-to-End Analytical Pipeline & Machine Learning Risk Classifier for Digital Lenders*

---

## 01. Executive Overview & Business Value

This repository contains the complete analytical pipeline, predictive model, and executive narrative designed for **Vitto**'s credit risk team. Using behavior and demographic data from **30,000 credit card clients** (UCI Credit Card Default Dataset, Taiwan, 2005), we identify the primary behavioral signals driving defaults and build a robust machine learning classifier.

### Key Finding
**Behavior trumps demographics.** A customer's repayment history and utilization rate over the last six months are exponentially more predictive of default than their gender, age, or education level.

* **Primary Signal (`TOTAL_DELAY_MONTHS`):** The cumulative number of months a borrower is late is the single most powerful warning signal, accounting for **45.8% of the total predictive weight** in our model.
* **Secondary Signal (`PAY_0`):** A borrower's payment status in the most recent month is the next strongest indicator. Customers currently 2+ months in arrears have a default rate of over **69%**.

---

## 02. Project Objective & Scope

The goal of this assessment is to build a credit decisioning asset that:
1. **Performs rigorous Exploratory Data Analysis (EDA)** to surface data anomalies, repayment delay heatmaps, and correlation patterns.
2. **Engineers behavior-driven features** representing credit utilization, repayment consistency, and delinquency duration.
3. **Trains and cross-validates** a baseline Logistic Regression and a high-performance XGBoost champion model.
4. **Conducts advanced audits** covering SQL-based querying, demographic fairness checks, and SHAP explainability.
5. **Provides concrete, non-technical business actions** for credit managers and loan officers.

---

## 03. Setup & Installation Guide

This project is built using Python 3.12+ and uses the high-performance **`uv`** package manager for fast, reliable virtual environments and dependency management.

### Prerequisites
Make sure you have Python 3.12+ and `uv` installed. If you do not have `uv`, you can install it via:
```bash
pip install uv
```

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd Credit-Default-Risk
   ```

2. **Initialize Environment & Install Dependencies:**
   Run the following command to automatically create a `.venv` virtual environment and sync all dependencies from `pyproject.toml` and `uv.lock`:
   ```bash
   uv sync
   ```

3. **Launch the Jupyter Notebook:**
   Start the Jupyter server in your environment to explore `Credit_Risk_Analysis.ipynb`:
   ```bash
   .venv\Scripts\jupyter notebook Credit_Risk_Analysis.ipynb
   ```

---

## 04. Feature Engineering Dictionary

We engineered three custom behavioral variables designed to capture borrower risk dynamics over a rolling 6-month period:

| Feature Name | Type | Formula / Logic | Business Rationale |
| :--- | :--- | :--- | :--- |
| **`AVG_UTIL_RATE`** | Numeric | `mean(BILL_AMT_x / LIMIT_BAL)` | Measures credit exposure. High utilization paired with low repayments indicates liquidity distress. |
| **`AVG_PAY_RATIO`** | Numeric | `mean(PAY_AMT_x / BILL_AMT_x)` (for `BILL_AMT_x > 0`) | Measures repayment consistency. Tracks what fraction of the outstanding bill the borrower actually pays off. |
| **`TOTAL_DELAY_MONTHS`**| Numeric | `count(PAY_x > 0)` | Delinquency duration. The total number of months in the past 6 months where the payment was 1+ months overdue. |

### Data Quality Transformations
* **`EDUCATION`:** Cleaned undocumented values (`0`, `5`, `6`) by merging them into category `4` ("Other").
* **`MARRIAGE`:** Cleaned undocumented value (`0`) by merging it into category `3` ("Other").
* **Categorical Encoding:** One-hot encoded `SEX`, `EDUCATION`, and `MARRIAGE` using `drop_first=True` to avoid the dummy variable trap.

---

## 05. Model Performance & Results

To handle the **22% class imbalance**, we optimized our models using stratified splitting, class weight balancing, and the Area Under the ROC Curve (**AUC-ROC**) metric.

### Model Evaluation Summary

| Metric | Logistic Regression (Baseline) | XGBoost (Champion Model) |
| :--- | :---: | :---: |
| **Precision (Class 1)** | 0.4600 | **0.4600** |
| **Recall (Class 1)** | 0.5700 | **0.5800** |
| **F1-Score (Class 1)** | 0.5100 | **0.5100** |
| **Test AUC-ROC** | 0.7501 | **0.7519** |
| **5-Fold CV AUC-ROC** | *N/A* | **0.7572 ± 0.0060** |

* **Generalization:** The extremely low standard deviation in our 5-fold cross-validation (`±0.0060`) proves the XGBoost model is stable and robust against overfitting.
* **Confusion Matrix Insight:** At our operating threshold, the model successfully flags **58% of actual defaults** (Recall) while maintaining a balanced precision, allowing Vitto's collections team to proactively mitigate risk.

---

## 06. Executive Summary & Concrete Actions

### Non-Technical Summary for Lending Managers
We analyzed six months of repayment behavior from 30,000 credit card clients to predict which customers are most likely to miss their next payment — before it happens. 

Our core finding is that **actions speak louder than demographics**. Gender, age, and education level have negligible predictive power compared to how a customer manages their card month to month. The single most powerful warning signal is the **total number of months a customer was late on payment** in the last six months. Accumulating three or more late months is a near-certain warning signal of impending default.

Additionally, the **most recent payment status** (September 2005) is the second strongest indicator. Customers who are already 2 or more months in arrears default at a rate of nearly **70%**. By utilizing our machine learning risk model, Vitto can identify approximately **58% of borrowers who will default next month**, ranking a high-risk borrower above a low-risk borrower **3 out of 4 times**—providing a solid baseline for a proactive credit intervention system.

### Two Concrete Credit Team Actions
1. **Early-Warning Delinquency Trigger:**  
   Implement a proactive outreach rule flagging any cardholder who records **2 or more delayed payment months** in a rolling 6-month window (based on `TOTAL_DELAY_MONTHS`). Soft interventions (SMS alerts, flexible repayment schedules) at this early stage are far more cost-effective than late-stage collection efforts or charge-offs.
2. **Behavioral Risk Limit Cap:**  
   Replace static annual reviews with a monthly behavioral score that combines utilization (`AVG_UTIL_RATE`) and payment ratios (`AVG_PAY_RATIO`). Any active borrower whose payment ratio drops below **10%** while utilization climbs past **80%** should have their credit limit automatically capped to mitigate Vitto's credit loss exposure.

---

## 07. Bonus Sections (Fully Implemented)

### I. SQL Business Analysis (SQLite)
We loaded the dataset into an in-memory SQL database and solved three core business questions:
* **Q1: Education Risk:** High school graduates exhibit the highest default rate (**25.16%**), followed closely by university graduates (**23.73%**).
* **Q2: Repayment Arrears:** Clients with a repayment delay of 2 months default at a rate of **69.14%**, rising to **71.92%** for 3+ months.
* **Q3: Financial Profile:** Defaulters have a significantly lower average credit limit (**130,110 NTD** vs. **178,100 NTD** for non-defaulters) and make substantially smaller payments (**3,397 NTD** vs. **6,307 NTD**).

### II. Algorithmic Fairness Audit
We audited the False Positive Rate (FPR) across demographics to ensure our credit decisioning asset does not unfairly discriminate:
* **FPR by Gender:** Males are flagged incorrectly at a higher rate than females (**22.4% vs 16.9%**), resulting in a **5.5 percentage point disparity**.
* **FPR by Education:** High school graduates have the highest false alarm rate (**23.1%**), while the "Other" category has the lowest (**2.6%**), showing a **20.5 pp maximum gap**.

### III. SHAP Model Explainability
We utilized **SHAP** values to decode model decisions:
* **Beeswarm Plot:** Explains global feature contributions, confirming that positive SHAP values (high default risk) are heavily driven by high `TOTAL_DELAY_MONTHS` and elevated `PAY_0` scores.
* **Waterfall Plots:** Provides individual, local explanations for 10 separate borrowers, showcasing exactly which features pushed their specific risk probability up or down—crucial for credit approval auditing.

---

## 08. Dataset & Citation

* **Source:** UCI Machine Learning Repository - [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
* **Citation:** Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2471-2475.
