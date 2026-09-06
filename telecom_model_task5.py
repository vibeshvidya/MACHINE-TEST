import csv
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
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


file_path = Path(__file__).resolve().parent / 'Churn.csv'
df = clean_data(load_data(file_path))

# Fit the churn model using the same encoded and scaled features as Task 3.
model_features = df.drop(columns=['Churn', 'customerID'], errors='ignore')
y = df['Churn'].map({'Yes': 1, 'No': 0})
X_encoded = pd.get_dummies(model_features, drop_first=True)
numeric_cols = X_encoded.select_dtypes(include=['number']).columns.tolist()

scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

X_train, _, y_train, _ = train_test_split(
	X_scaled,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y,
)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
df['PredictedChurnProbability'] = model.predict_proba(X_scaled)[:, 1]

# Cluster the three targeting variables after putting them on comparable scales.
segmentation_features = ['tenure', 'MonthlyCharges', 'PredictedChurnProbability']
segment_scaler = StandardScaler()
segmentation_values = segment_scaler.fit_transform(df[segmentation_features])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
df['Segment'] = kmeans.fit_predict(segmentation_values) + 1

summary = (
	df.groupby('Segment', as_index=False)
	  .agg(
		  Customers=('Segment', 'size'),
		  AverageTenure=('tenure', 'mean'),
		  AverageMonthlyCharges=('MonthlyCharges', 'mean'),
		  AverageChurnRisk=('PredictedChurnProbability', 'mean'),
	  )
)

tenure_median = summary['AverageTenure'].median()
spend_median = summary['AverageMonthlyCharges'].median()
risk_median = summary['AverageChurnRisk'].median()


def make_segment_name(row):
	tenure_name = 'New' if row['AverageTenure'] <= tenure_median else 'Established'
	spend_name = 'High-Spend' if row['AverageMonthlyCharges'] >= spend_median else 'Low-Spend'
	risk_name = 'High-Risk' if row['AverageChurnRisk'] >= risk_median else 'Low-Risk'
	return f'{tenure_name}, {spend_name}, {risk_name}'


summary['BusinessName'] = summary.apply(make_segment_name, axis=1)
df = df.merge(summary[['Segment', 'BusinessName']], on='Segment', how='left')

print('=== Customer Targeting Segments ===')
print(summary.sort_values('AverageChurnRisk', ascending=False).round({
	'AverageTenure': 2,
	'AverageMonthlyCharges': 2,
	'AverageChurnRisk': 4,
}).to_string(index=False))

output_columns = [
	col for col in [
		'customerID', 'Segment', 'BusinessName', 'tenure', 'MonthlyCharges',
		'PredictedChurnProbability'
	] if col in df.columns
]
output_path = file_path.parent / 'customer_segments.csv'
df[output_columns].to_csv(output_path, index=False)
print(f'\nSaved customer-level segments to: {output_path}')