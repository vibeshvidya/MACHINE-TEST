import csv
import json
import os
from pathlib import Path
from urllib import error, request

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PLAYBOOK_CLAUSES = {
    1: 'For a new high-risk customer, offer a 12- or 24-month contract with temporary price protection and schedule a service check-in within 30 days.',
    2: 'For an established high-risk customer, offer a loyalty discount tied to a longer contract and proactively review the customer\'s current plan.',
    3: 'For a new moderate-risk customer, offer an introductory loyalty bundle and a follow-up review before the first annual renewal.',
    4: 'For an established moderate-risk customer, offer a loyalty add-on bundle and review plan value at the next renewal.',
}
FORBIDDEN_TERMS = ('gender', 'SeniorCitizen', 'Partner', 'Dependents')


def load_data(file_path):
    with file_path.open('r', encoding='utf-8-sig', newline='') as file:
        rows = [row for row in csv.reader(file)]

    if rows and len(rows[0]) == 1:
        with file_path.open('r', encoding='utf-8-sig', newline='') as file:
            lines = [line.strip() for line in file if line.strip()]
        rows = [line.strip('"').split(',') for line in lines]

    if not rows:
        raise ValueError('The dataset is empty.')

    header = rows[0]
    clean_rows = [row for row in rows[1:] if len(row) == len(header)]
    return pd.DataFrame(clean_rows, columns=header)


def clean_data(df):
    for col in ['PaymentMethod', 'PaperlessBilling']:
        if col in df.columns:
            df = df.drop(columns=[col])

    text_cols = [
        'Churn', 'Contract', 'InternetService', 'gender', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines',
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    df['Churn'] = df['Churn'].str.replace(' ', '', regex=False)
    df = df[df['Churn'].isin(['Yes', 'No'])].copy()

    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in df.columns:
        if col in ['Churn', 'customerID']:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])

    return df.dropna(subset=['Churn', 'tenure', 'MonthlyCharges']).reset_index(drop=True)


@st.cache_resource
def train_churn_model(file_path):
    data = clean_data(load_data(file_path))
    model_features = data.drop(columns=['Churn', 'customerID'], errors='ignore')
    target = data['Churn'].map({'Yes': 1, 'No': 0})
    encoded = pd.get_dummies(model_features, drop_first=True)
    numeric_cols = encoded.select_dtypes(include=['number']).columns.tolist()
    scaler = StandardScaler()
    scaled = encoded.copy()
    scaled[numeric_cols] = scaler.fit_transform(encoded[numeric_cols])

    X_train, _, y_train, _ = train_test_split(
        scaled, target, test_size=0.2, random_state=42, stratify=target,
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    defaults = model_features.mode().iloc[0].to_dict()
    return model, scaler, scaled.columns.tolist(), defaults


def predict_customer(model, scaler, feature_columns, defaults, customer):
    row = defaults.copy()
    row.update(customer)
    encoded = pd.get_dummies(pd.DataFrame([row]), drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    numeric_cols = encoded.select_dtypes(include=['number']).columns.tolist()
    scaled = encoded.copy()
    scaled[numeric_cols] = scaler.transform(encoded[numeric_cols])
    probability = float(model.predict_proba(scaled)[0, 1])
    contributions = pd.Series(
        model.coef_[0] * scaled.iloc[0], index=feature_columns,
    ).sort_values(ascending=False)
    top_features = [
        feature for feature in contributions.index
        if not any(term.lower() in feature.lower() for term in FORBIDDEN_TERMS)
    ][:3]
    return probability, top_features


def retrieve_playbook_clause(risk_probability, tenure):
    # This function is deliberately called inside the request handler below.
    if risk_probability >= 0.5 and tenure < 24:
        return 1, PLAYBOOK_CLAUSES[1]
    if risk_probability >= 0.5:
        return 2, PLAYBOOK_CLAUSES[2]
    if tenure < 24:
        return 3, PLAYBOOK_CLAUSES[3]
    return 4, PLAYBOOK_CLAUSES[4]


def call_gemini(system_prompt, user_prompt):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('GEMINI_API_KEY is required to generate an explanation.')

    payload = json.dumps({
        'system_instruction': {'parts': [{'text': system_prompt}]},
        'contents': [{'role': 'user', 'parts': [{'text': user_prompt}]}],
        'generationConfig': {'temperature': 0},
    }).encode('utf-8')
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
    http_request = request.Request(
        f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with request.urlopen(http_request, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
    except error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Gemini request failed ({exc.code}): {detail}') from exc

    try:
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f'Gemini returned an unexpected response: {result}') from exc


st.set_page_config(page_title='Telecom Retention Advisor', page_icon='R', layout='centered')
st.title('Telecom Retention Advisor')
st.caption('Enter customer details to generate a grounded retention recommendation.')

file_path = Path(__file__).resolve().parent / 'Churn.csv'
model, scaler, feature_columns, defaults = train_churn_model(file_path)

with st.form('customer_form'):
    st.subheader('Customer details')
    customer_id = st.text_input('Customer ID', value='manual-entry')
    tenure = st.number_input('Tenure (months)', min_value=0.0, max_value=1000.0, value=12.0, step=1.0)
    monthly_charges = st.number_input('Monthly charges', min_value=0.0, value=75.0, step=1.0)
    total_charges = st.number_input('Total charges', min_value=0.0, value=900.0, step=10.0)
    contract = st.selectbox('Contract', ['Month-To-Month', 'One Year', 'Two Year'])
    internet_service = st.selectbox('Internet service', ['Fiber Optic', 'Dsl', 'No'])
    submitted = st.form_submit_button('Assess retention risk')

if submitted:
    customer = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
    }
    try:
        risk_probability, top_features = predict_customer(
            model, scaler, feature_columns, defaults, customer,
        )
        risk_tier = 'High risk' if risk_probability >= 0.5 else 'Moderate risk'

        # Retrieval runs for every submitted request; it is not cached at startup.
        clause_number, clause_text = retrieve_playbook_clause(risk_probability, tenure)
        system_prompt = (
            'You are a telecom retention advisor writing for a retention agent. Produce exactly '
            '3 or 4 sentences. Use only the retrieved clause and supplied feature names as '
            'evidence. Do not invent facts or actions. Never mention gender, SeniorCitizen, '
            'Partner, Dependents, or any proxy or implication of them.'
        )
        user_prompt = (
            f'Retrieved Clause {clause_number}: {clause_text}\n'
            f'Risk probability: {risk_probability:.4f}\n'
            f'Tenure: {tenure:.0f} months\n'
            f'Top contributing feature names: {", ".join(top_features)}'
        )
        explanation = call_gemini(system_prompt, user_prompt)

        st.success('Assessment complete')
        st.metric('Risk tier', risk_tier)
        st.metric('Predicted churn probability', f'{risk_probability:.1%}')
        st.subheader(f'Cited playbook Clause {clause_number}')
        st.info(clause_text)
        st.subheader('LLM explanation')
        st.write(explanation)
        st.caption(f'Customer: {customer_id} | Retrieval executed for this request')
    except RuntimeError as exc:
        st.error(str(exc))
