import csv
from pathlib import Path

import pandas as pd

file_path = Path(__file__).resolve().parent / 'Churn.csv'

# Load the raw CSV safely. The file has one quoted row per line, so avoid a plain pd.read_csv() here.
with file_path.open('r', encoding='utf-8-sig', newline='') as f:
    rows = [row for row in csv.reader(f)]

# If pandas read the file as a single-column table, split each text row manually.
if rows and len(rows[0]) == 1:
    with file_path.open('r', encoding='utf-8-sig', newline='') as f:
        lines = [line.strip() for line in f if line.strip()]
    rows = [line.strip('"').split(',') for line in lines]

header = rows[0]
expected_len = len(header)
clean_rows = [row for row in rows[1:] if len(row) == expected_len]
df = pd.DataFrame(clean_rows, columns=header)

# Drop unnecessary columns for modeling.
# customerID is just an identifier and not useful as a predictive feature.
# PaymentMethod and PaperlessBilling are not required for this task and can be removed to simplify the model.
for col in ['customerID', 'PaymentMethod', 'PaperlessBilling']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Remove malformed rows with mismatched column lengths.
# Keeping them would create misaligned values and break feature engineering.
print(f'Rows before cleaning: {len(df)}')

# Fix inconsistent text values.
for col in ['Churn', 'Contract', 'InternetService', 'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# Clean the target variable.
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].str.replace(' ', '', regex=False)
    df = df[df['Churn'].isin(['Yes', 'No'])].copy()

# Convert numeric columns to numeric, treating blanks as missing.
for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values.
# For continuous columns, median imputation preserves the central tendency without being distorted by outliers.
# For categorical columns, mode imputation preserves the most common category.
for col in df.columns:
    if col == 'Churn':
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        mode_value = df[col].mode()
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value.iloc[0])

# Drop rows with missing target or key fields that cannot be reasonably imputed.
# These rows are not usable for churn modeling because the target is unavailable or essential business context is missing.
df = df.dropna(subset=['Churn', 'Contract', 'InternetService', 'tenure']).reset_index(drop=True)

# Prepare features and target for modeling.
# Encode categorical features and scale numeric features before modeling.
X = df.drop(columns=['Churn'])
y = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode categorical variables.
X_encoded = pd.get_dummies(X, drop_first=True)

# Identify numeric columns for scaling.
numeric_cols = X_encoded.select_dtypes(include=['number']).columns.tolist()

# Scale numeric features to comparable ranges for algorithms sensitive to feature scale.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

print(f'Rows after cleaning: {len(df)}')
print(f'Feature matrix shape before scaling: {X_encoded.shape}')
print(f'Feature matrix shape after scaling: {X_scaled.shape}')
print(f'Target shape: {y.shape}')
print('\nPreview of cleaned dataset:')
print(df.head().to_string(index=False))

# Optional: print the first few scaled feature columns
print('\nScaled feature sample:')
print(X_scaled.head().to_string(index=False))

# Overall churn KPI
overall_churn_rate = df['Churn'].eq('Yes').mean() * 100
print(f'\nOverall churn rate: {overall_churn_rate:.2f}%')

# Churn rate by contract type
contract_churn = (
    df.groupby('Contract')['Churn']
      .apply(lambda s: (s == 'Yes').mean() * 100)
      .sort_values(ascending=False)
      .rename('ChurnRate (%)')
)
print('\nChurn rate by Contract type:')
print(contract_churn.round(2).to_string())

# Churn rate by internet service type
internet_churn = (
    df.groupby('InternetService')['Churn']
      .apply(lambda s: (s == 'Yes').mean() * 100)
      .sort_values(ascending=False)
      .rename('ChurnRate (%)')
)
print('\nChurn rate by InternetService type:')
print(internet_churn.round(2).to_string())

# Tenure vs Churn correlation
churn_numeric = df['Churn'].map({'Yes': 1, 'No': 0})
tenure_churn_corr = df['tenure'].corr(churn_numeric)
print(f'\nCorrelation between tenure and churn: {tenure_churn_corr:.4f}')

avg_tenure_churned = df.loc[churn_numeric == 1, 'tenure'].mean()
avg_tenure_retained = df.loc[churn_numeric == 0, 'tenure'].mean()
print(f'Average tenure of churned customers: {avg_tenure_churned:.2f} months')
print(f'Average tenure of retained customers: {avg_tenure_retained:.2f} months')


#Part 2: Predictive Modelling & Business Segmentation 
#Task 2: Clean & Preprocess

