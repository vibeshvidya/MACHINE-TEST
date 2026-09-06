import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    with file_path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = [row for row in csv.reader(f)]

    if rows and len(rows[0]) == 1:
        with file_path.open('r', encoding='utf-8-sig', newline='') as f:
            lines = [line.strip() for line in f if line.strip()]
        rows = [line.strip('"').split(',') for line in lines]

    if not rows:
        raise ValueError('The dataset is empty.')

    header = rows[0]
    clean_rows = [row for row in rows[1:] if len(row) == len(header)]
    return pd.DataFrame(clean_rows, columns=header)


def clean_and_prepare_data(df):
    for col in ['customerID', 'PaymentMethod', 'PaperlessBilling']:
        if col in df.columns:
            df = df.drop(columns=[col])

    text_cols = [
        'Churn', 'Contract', 'InternetService', 'gender', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines'
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].str.replace(' ', '', regex=False)
        df = df[df['Churn'].isin(['Yes', 'No'])].copy()

    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in df.columns:
        if col == 'Churn':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])

    df = df.dropna(subset=['Churn', 'Contract', 'InternetService', 'tenure']).reset_index(drop=True)
    return df


file_path = Path(__file__).resolve().parent / 'Churn.csv'
df = load_data(file_path)
df = clean_and_prepare_data(df)

X = df.drop(columns=['Churn'])
y = df['Churn'].map({'Yes': 1, 'No': 0})

X_encoded = pd.get_dummies(X, drop_first=True)
num_cols = X_encoded.select_dtypes(include=['number']).columns.tolist()

scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[num_cols] = scaler.fit_transform(X_encoded[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

coef_df = pd.DataFrame({
    'feature': X_scaled.columns,
    'coef': model.coef_[0],
})
coef_df['abs_coef'] = coef_df['coef'].abs()

important_features = coef_df.sort_values('abs_coef', ascending=False).head(10)

print('=== Best model summary ===')
print(f'Training Accuracy: {train_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print('\nTop 10 feature importance values:')
print(important_features[['feature', 'coef']].to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 7))
colors = ['tab:red' if coef >= 0 else 'tab:blue' for coef in important_features['coef']]
plt.barh(important_features['feature'][::-1], important_features['coef'][::-1], color=colors[::-1])
plt.axvline(0, color='black', linewidth=1)
plt.title('Top 10 Features Driving Churn Prediction (Logistic Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plot_path = file_path.parent / 'feature_importance.png'
plt.savefig(plot_path, dpi=300)
print(f'\nSaved feature importance plot to: {plot_path}')

