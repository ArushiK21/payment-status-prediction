# Payment Status Prediction using Logistic Regression

This project applies **logistic regression** to predict payment status (paid / not paid) based on customer features.  
It demonstrates data encoding, model training/testing, and interpretation of results using **odds ratios**.

---

## Files in this Repository
- `data/logistic_regression_dataset.csv` – Original dataset  
- `scripts/encoding_logistic_regression.py` – Data encoding (binary + ordinal)  
- `scripts/train_test_logistic_regression.py` – Train/test split and logistic regression model  
- `scripts/odds_ratio_logistic_regression.py` – Odds ratio interpretation  
- `results/encoded_dataset.csv` – Encoded dataset  
- `results/predictions.csv` – Actual vs Predicted payment status  
- `results/odds_ratios.csv` – Odds ratio analysis of features  
- `reports/Payment_Status_Analysis_Report.docx` – Detailed project report  
- `reports/Payment_Status_Analysis_Presentation.pptx` – Presentation slides  

---

## How to Run the Code


### 1. Install dependencies
pip install pandas numpy scikit-learn statsmodels category_encoders

### 2. Run scripts
- Encoding dataset
python scripts/encoding_logistic_regression.py

- Training & testing
python scripts/train_test_logistic_regression.py

- Odds ratio interpretation
python scripts/odds_ratio_logistic_regression.py

---

## Key Findings

Data Preparation: Applied binary encoding for gender, marital status, city, and province; ordinal encoding for education and age range.

Model Accuracy: Achieved accuracy on test data but observed overfitting (model performed better on training data).

Feature Impact (Odds Ratio):

📈 Features increasing odds of non-payment:

Appcount (1.31) → each additional application increases odds of non-payment by 31%.

Education (1.20) → higher education slightly increases default risk.

City_2 (1.21) → living in certain cities increases non-payment odds.

📉 Features reducing odds of non-payment:

FICO Score (0.58) → higher credit score reduces odds of non-payment by 42%.

Salary (0.67) → higher income reduces odds by 33%.

Married Status (0.68) → being married lowers risk.

---

## Project Workflow

Step 1: Encode categorical variables (binary & ordinal encoding).

Step 2: Train logistic regression model (90% train, 10% test).

Step 3: Evaluate model using accuracy, confusion matrix, and classification report.

Step 4: Interpret model coefficients using odds ratios. 

---

## Notes 

This project was originally completed as an academic project in Predictive Analytics & Empirical Finance.

