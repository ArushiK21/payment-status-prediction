import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the data
file_path = r"C:\Users\user\OneDrive\Desktop\github upload\encoded_dataset.csv"
df = pd.read_csv(file_path)

# Step 2: Define target and features
target_column = 'PaymentStatus'
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Step 4: Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Convert scaled data back to DataFrame to retain column names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_scaled_df = sm.add_constant(X_train_scaled_df)

# Align indices
y_train = y_train.reset_index(drop=True)

# Step 6: Fit logistic regression using statsmodels for named summary
logit_model = sm.Logit(y_train, X_train_scaled_df)
result = logit_model.fit()

# Step 7: Show detailed regression summary
print(result.summary())

# Step 8: Predict and evaluate
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
X_test_scaled_df = sm.add_constant(X_test_scaled_df)

y_pred = result.predict(X_test_scaled_df)
y_pred_class = (y_pred >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred_class))

# Step 9: Save results
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_class
}).reset_index(drop=True)

output_path = r"C:\Users\user\Downloads\predictions.csv"
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")



