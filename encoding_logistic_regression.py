import pandas as pd
import category_encoders as ce

# Load your dataset
df = pd.read_csv(r"C:\Users\user\Downloads\logistic_regression_dataset_ (Orginal File).csv")  # Replace with your actual file path

# Ordinal encoding for Education and AgeRange
ordinal_encoder = ce.OrdinalEncoder(cols=['Education', 'AgeRange'])
df = ordinal_encoder.fit_transform(df)

# Encode Gender and Married_Status using binary mapping
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Married_Status'] = df['Married_Status'].map({'Single': 0, 'Married': 1})

# Binary encoding for City and Province
binary_encoder = ce.BinaryEncoder(cols=['City', 'Province'])
df = binary_encoder.fit_transform(df)

# Preview the encoded dataframe
print(df.head())


# Save to new Excel file for easy observation
df.to_csv(r"C:\Users\user\Downloads\encoded_dataset.csv", index=False)

