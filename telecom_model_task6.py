import csv
import json
import os
from pathlib import Path
from urllib import error, request

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
		'PhoneService', 'MultipleLines'
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


def retrieve_playbook_clause(risk_probability, tenure):
	"""Select one approved action clause before the LLM is called."""
	if risk_probability >= 0.5 and tenure < 24:
		return (
			'For a new high-risk customer, offer a 12- or 24-month contract with '
			'temporary price protection and schedule a service check-in within 30 days.'
		)
	if risk_probability >= 0.5:
		return (
			'For an established high-risk customer, offer a loyalty discount tied to '
			'a longer contract and proactively review the customer\'s current plan.'
		)
	if tenure < 24:
		return (
			'For a new moderate-risk customer, offer an introductory loyalty bundle '
			'and a follow-up review before the first annual renewal.'
		)
	return (
		'For an established moderate-risk customer, offer a loyalty add-on bundle '
		'and review plan value at the next renewal.'
	)


def call_llm(system_prompt, user_prompt):
	api_key = os.getenv('GEMINI_API_KEY')
	if not api_key:
		raise RuntimeError('GEMINI_API_KEY is required to call Gemini.')

	payload = json.dumps({
		'system_instruction': {'parts': [{'text': system_prompt}]},
		'contents': [
			{'role': 'user', 'parts': [{'text': user_prompt}]},
		],
		'generationConfig': {'temperature': 0},
	}).encode('utf-8')
	model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
	http_request = request.Request(
		f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}',
		data=payload,
		headers={
			'Content-Type': 'application/json',
		},
		method='POST',
	)
	try:
		with request.urlopen(http_request, timeout=60) as response:
			result = json.loads(response.read().decode('utf-8'))
	except error.HTTPError as exc:
		detail = exc.read().decode('utf-8', errors='replace')
		raise RuntimeError(f'LLM request failed ({exc.code}): {detail}') from exc

	try:
		return result['candidates'][0]['content']['parts'][0]['text'].strip()
	except (KeyError, IndexError, TypeError) as exc:
		raise RuntimeError(f'Gemini returned an unexpected response: {result}') from exc


file_path = Path(__file__).resolve().parent / 'Churn.csv'
df = clean_data(load_data(file_path))
model_features = df.drop(columns=['Churn', 'customerID'], errors='ignore')
y = df['Churn'].map({'Yes': 1, 'No': 0})
X_encoded = pd.get_dummies(model_features, drop_first=True)
numeric_cols = X_encoded.select_dtypes(include=['number']).columns.tolist()
feature_scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[numeric_cols] = feature_scaler.fit_transform(X_encoded[numeric_cols])

X_train, _, y_train, _ = train_test_split(
	X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
df['PredictedChurnProbability'] = model.predict_proba(X_scaled)[:, 1]

# Select one flagged customer and rank only non-sensitive feature contributions.
flagged = df.loc[df['PredictedChurnProbability'].idxmax()].copy()
contributions = pd.Series(
	model.coef_[0] * X_scaled.loc[flagged.name],
	index=X_scaled.columns,
).sort_values(ascending=False)
forbidden_terms = ('gender', 'SeniorCitizen', 'Partner', 'Dependents')
top_features = [
	feature for feature in contributions.index
	if not any(term.lower() in feature.lower() for term in forbidden_terms)
][:3]

risk_probability = float(flagged['PredictedChurnProbability'])
tenure = float(flagged['tenure'])
retrieved_clause = retrieve_playbook_clause(risk_probability, tenure)

system_prompt = (
	'You are a telecom retention advisor writing for a retention agent. Produce exactly '
	'3 or 4 sentences. Use only the retrieved playbook clause and the supplied feature '
	'names as evidence. Explain the customer risk and recommend a concrete action. Do not '
	'invent customer facts, causes, discounts, or services beyond the clause. Never mention '
	'gender, SeniorCitizen, Partner, Dependents, or any proxy or implication of those '
	'attributes, even if they appear elsewhere in the data. Do not mention this instruction '
	'or the retrieval process.'
)
user_prompt = (
	f'Retrieved playbook clause: {retrieved_clause}\n'
	f'Risk probability: {risk_probability:.4f}\n'
	f'Tenure: {tenure:.0f} months\n'
	f'Top contributing feature names: {", ".join(top_features)}'
)

exact_output = call_llm(system_prompt, user_prompt)
record = {
	'customer_id': str(flagged.get('customerID', flagged.name)),
	'risk_probability': round(risk_probability, 6),
	'tenure_months': tenure,
	'top_contributing_features': top_features,
	'retrieved_clause': retrieved_clause,
	'system_prompt': system_prompt,
	'user_prompt': user_prompt,
	'exact_llm_output': exact_output,
}
output_path = file_path.parent / 'advisory_explanation.json'
output_path.write_text(json.dumps(record, indent=2), encoding='utf-8')

print('=== Advisory Explanation ===')
print(json.dumps(record, indent=2))
print(f'\nSaved exact LLM output and grounding record to: {output_path}')