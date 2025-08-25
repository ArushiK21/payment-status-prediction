import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# STEP 1: Load your 29th assignment excel path here
df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\github upload\encoded_dataset.csv")

# STEP 2: Define features (X) and target (y)
X = df.drop("PaymentStatus", axis=1)
y = df["PaymentStatus"]

# STEP 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# STEP 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# STEP 5: Fit logistic regression
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# STEP 6: Interpret coefficients using odds ratio
feature_names = X_train.columns
coefficients = model.coef_[0]
odds_ratios = np.exp(coefficients)

odds_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
}).sort_values(by='Odds Ratio', ascending=False)

print(odds_df)

# STEP 7: Save to new Excel file for easy observation
odds_df.to_csv(r"C:\Users\user\Downloads\odds_ratios.csv", index=False)

