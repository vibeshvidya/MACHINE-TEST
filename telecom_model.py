from pathlib import Path

import pandas as pd

file_path = Path(__file__).resolve().parent / 'Churn.csv'

with file_path.open('r', encoding='utf-8-sig') as f:
    lines = [line.strip() for line in f if line.strip()]

# The file is stored as one quoted row per line, so we strip quotes and split each row manually.
rows = [line.strip('"').split(',') for line in lines]
header = rows[0]
values = rows[1:]
df = pd.DataFrame(values, columns=header)

# Standardize text values
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].astype(str).str.strip().str.title()

if 'Contract' in df.columns:
    df['Contract'] = df['Contract'].astype(str).str.strip()

if 'InternetService' in df.columns:
    df['InternetService'] = df['InternetService'].astype(str).str.strip()

# Convert data types
if 'tenure' in df.columns:
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

# Overall churn KPI
overall_churn_rate = df['Churn'].eq('Yes').mean() * 100
print(f'Overall churn rate: {overall_churn_rate:.2f}%')

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

