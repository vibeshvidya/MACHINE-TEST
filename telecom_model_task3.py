import csv
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
    df = pd.DataFrame(clean_rows, columns=header)

    return df


def clean_and_prepare_data(df):
    # Drop unnecessary columns
    for col in ['customerID', 'PaymentMethod', 'PaperlessBilling']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Standardize text fields
    text_cols = [
        'Churn', 'Contract', 'InternetService', 'gender', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines'
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Keep only valid churn labels
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].str.replace(' ', '', regex=False)
        df = df[df['Churn'].isin(['Yes', 'No'])].copy()

    # Convert numeric columns
    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values
    for col in df.columns:
        if col == 'Churn':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])

    # Drop rows with missing target or critical info
    df = df.dropna(subset=['Churn', 'Contract', 'InternetService', 'tenure']).reset_index(drop=True)

    return df


# Load and preprocess the data
file_path = Path(__file__).resolve().parent / 'Churn.csv'
df = load_data(file_path)
df = clean_and_prepare_data(df)

# Prepare features and target
X = df.drop(columns=['Churn'])
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical variables and scale numeric variables
X_encoded = pd.get_dummies(X, drop_first=True)
numeric_cols = X_encoded.select_dtypes(include=['number']).columns.tolist()

scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
y_train_pred = model.predict(X_train)

# Evaluation metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print('=== Churn Prediction Model Summary ===')
print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print('\nConfusion Matrix:')
print(cm)
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Trust / interpretability check
coef_df = pd.DataFrame({
    'feature': X_scaled.columns,
    'coef': model.coef_[0]
}).sort_values(by='coef', ascending=False)

print('\nTop positive coefficients (higher value means more likely to churn):')
print(coef_df.head(10).to_string(index=False))

print('\nTop negative coefficients (higher value means less likely to churn):')
print(coef_df.tail(10).sort_values(by='coef', ascending=True).to_string(index=False))

# Business interpretation
TN, FP, FN, TP = cm.ravel()
print(f'\nTrue Negatives: {TN}')
print(f'False Positives: {FP}')
print(f'False Negatives: {FN}')
print(f'True Positives: {TP}')

# Save final prepared data for reference
print('\nPrepared dataset shape:', df.shape)
print('Model-ready feature matrix shape:', X_scaled.shape)