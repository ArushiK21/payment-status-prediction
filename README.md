# **Payment Status Prediction using Logistic Regression**

This project applies **logistic regression** to predict customer payment status as **paid / not paid** based on demographic, financial, and application-related features.

The project demonstrates the full predictive analytics workflow, including:

* Categorical data encoding
* Train-test splitting
* Feature scaling
* Logistic regression model training
* Model evaluation
* Odds ratio interpretation
* Exporting predictions and results

The project was completed as part of an academic assignment in **Empirical Finance & Predictive Analysis**.

---

## **1. Project Overview**

The goal of this project is to build an interpretable machine learning model that predicts whether a customer is likely to be classified as a non-payer.

Logistic regression was selected because it is suitable for binary classification problems and provides interpretable coefficients. These coefficients can be converted into **odds ratios**, which help explain how each feature affects the likelihood of non-payment.

---

## **2. Repository Structure**

```text
Payment-Status-Prediction/
│
├── data/
│   └── logistic_regression_dataset.csv
│
├── scripts/
│   ├── encoding_logistic_regression.py
│   ├── train_test_logistic_regression.py
│   └── odds_ratio_logistic_regression.py
│
├── results/
│   ├── encoded_dataset.csv
│   ├── predictions.csv
│   └── odds_ratios.csv
│
├── reports/
│   ├── Payment_Status_Analysis_Report.docx
│   └── Payment_Status_Analysis_Presentation.pptx
│
└── README.md
```

---

## **3. Files in This Repository**

| File                                                | Description                                                     |
| --------------------------------------------------- | --------------------------------------------------------------- |
| `data/logistic_regression_dataset.csv`              | Original dataset used for the analysis                          |
| `scripts/encoding_logistic_regression.py`           | Encodes categorical variables using binary and ordinal encoding |
| `scripts/train_test_logistic_regression.py`         | Trains and evaluates the logistic regression model              |
| `scripts/odds_ratio_logistic_regression.py`         | Converts model coefficients into odds ratios                    |
| `results/encoded_dataset.csv`                       | Final encoded dataset                                           |
| `results/predictions.csv`                           | Actual vs predicted payment status                              |
| `results/odds_ratios.csv`                           | Odds ratio interpretation of model features                     |
| `reports/Payment_Status_Analysis_Report.docx`       | Detailed academic project report                                |
| `reports/Payment_Status_Analysis_Presentation.pptx` | Project presentation slides                                     |

---

## **4. Dataset and Features**

The dataset contains customer-level information used to predict payment status.

The target variable is:

```text
PaymentStatus
```

where the model predicts whether a customer is likely to be classified as a payer or non-payer.

The feature set includes variables such as:

* Age
* Gender
* Education
* Marital status
* City
* Province
* Age range
* Salary
* FICO score
* Application count
* Phone grade

---

## **5. Methodology**

The project follows three main stages.

### **5.1 Data Encoding**

Categorical variables were converted into numerical form to make them usable for logistic regression.

The encoding strategy included:

* **Binary encoding** for variables such as gender, marital status, city, and province
* **Ordinal encoding** for ordered variables such as education level and age range
* Keeping numerical variables such as salary, FICO score, application count, and phone grade in numerical form

This step created a clean encoded dataset for model training.

---

### **5.2 Model Training and Testing**

A logistic regression model was trained using a **90% training and 10% testing split**.

The workflow included:

1. Loading the encoded dataset
2. Separating features and target variable
3. Splitting the data into training and testing sets
4. Standardizing numerical features
5. Training the logistic regression model
6. Predicting payment status on the test set
7. Evaluating model performance using:

   * Accuracy score
   * Confusion matrix
   * Classification report

---

### **5.3 Odds Ratio Interpretation**

To interpret the impact of each feature, logistic regression coefficients were converted into odds ratios.

The odds ratio helps explain whether a variable increases or decreases the odds of non-payment.

General interpretation:

* **Odds ratio > 1** means the feature increases the odds of non-payment
* **Odds ratio < 1** means the feature decreases the odds of non-payment
* **Odds ratio = 1** means the feature has no meaningful effect

---

## **6. Key Findings**

### **6.1 Features Increasing the Odds of Non-Payment**

The following variables increased the odds of non-payment:

| Feature   | Odds Ratio | Interpretation                                                                     |
| --------- | ---------: | ---------------------------------------------------------------------------------- |
| Appcount  |       1.31 | Each additional application increases the odds of non-payment by approximately 31% |
| Education |       1.20 | Higher education level is associated with a slight increase in non-payment odds    |
| City_2    |       1.21 | Customers from this city group show slightly higher non-payment risk               |
| City_0    |       1.14 | Customers from this city group also show slightly higher non-payment risk          |

---

### **6.2 Features Reducing the Odds of Non-Payment**

The following variables reduced the odds of non-payment:

| Feature        | Odds Ratio | Interpretation                                                         |
| -------------- | ---------: | ---------------------------------------------------------------------- |
| FICO Score     |       0.58 | Higher FICO score reduces the odds of non-payment by approximately 42% |
| Salary         |       0.67 | Higher salary reduces the odds of non-payment by approximately 33%     |
| Married Status |       0.68 | Being married appears to reduce the risk of non-payment                |

---

## **7. Model Performance and Interpretation**

The logistic regression model provides a useful baseline for predicting payment status and understanding feature impact.

However, the model also shows signs of limited generalization. While the overall accuracy appears acceptable, the classification results suggest that the model performs better for the majority class than for the minority class.

This means the model is useful as an interpretable baseline, but it should not be treated as a final production-ready credit risk model without further improvements.

---

## **8. Model Limitations**

This project has the following limitations:

* The dataset appears to be imbalanced, which affects minority-class prediction.
* Accuracy alone may not fully represent model quality.
* The model may overfit the training data.
* Logistic regression assumes a linear relationship between features and log-odds.
* More advanced models could be tested for comparison.
* Additional evaluation metrics such as precision, recall, F1-score, ROC-AUC, and PR-AUC would improve the analysis.

---

## **9. How to Run the Code**

### **Step 1: Install Dependencies**

```bash
pip install pandas numpy scikit-learn statsmodels category_encoders
```

### **Step 2: Run Data Encoding**

```bash
python scripts/encoding_logistic_regression.py
```

### **Step 3: Train and Test the Logistic Regression Model**

```bash
python scripts/train_test_logistic_regression.py
```

### **Step 4: Generate Odds Ratio Interpretation**

```bash
python scripts/odds_ratio_logistic_regression.py
```

---

## **10. Tools and Libraries Used**

* Python
* pandas
* NumPy
* scikit-learn
* statsmodels
* category_encoders
* Excel / CSV files

---

## **11. Possible Future Improvements**

Future improvements could include:

* Handling class imbalance using class weights, oversampling, or SMOTE
* Comparing logistic regression with decision trees, random forest, or gradient boosting
* Adding ROC-AUC and precision-recall curve analysis
* Performing cross-validation
* Improving feature engineering
* Testing regularized logistic regression
* Creating a simple dashboard for model outputs

---

## **12. Conclusion**

This project demonstrates how logistic regression can be used for payment status prediction and credit-risk-style classification.

The strongest part of the project is its interpretability. Through odds ratios, the model shows which features increase or reduce the likelihood of non-payment. Variables such as application count, education, and city group increase non-payment risk, while FICO score, salary, and marital status reduce it.

Although the model provides a useful baseline, the evaluation results suggest that further improvements are needed to handle class imbalance and improve minority-class prediction.

---

## **13. Disclaimer**

This project is for academic and educational purposes only. It is not intended for real-world lending, credit approval, or financial decision-making without further validation.


