"""
Script to train and save the churn prediction model
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Starting model training...")

# Define features that will be used for prediction
categorical_features = [
    'Gender', 'Location', 'Plan_Type'
]

numerical_features = [
    'Age', 'Tenure_Months', 'Data_Usage_GB', 'Call_Minutes', 
    'SMS_Count', 'Payment_History_12mo', 'Outstanding_Balance',
    'Monthly_Bill', 'Num_Complaints', 'Support_Tickets', 'Payment_Delays'
]

# Load the sample data
try:
    print("Loading sample data...")
    df = pd.read_csv('attached_assets/netone_customers.csv')
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit(1)

# Preprocessing
print("Preprocessing data...")
# Keep only required columns for prediction
columns_to_keep = numerical_features + categorical_features + ['Churn']
available_columns = [col for col in columns_to_keep if col in df.columns]
if len(available_columns) < len(columns_to_keep):
    missing = set(columns_to_keep) - set(available_columns)
    print(f"Warning: Missing columns: {missing}")

df = df[available_columns].copy()

# Check for and handle missing values
for col in numerical_features:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(float)

for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Handle the target column
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].fillna(0).astype(int)
else:
    print("Error: Churn column not found in the data. Cannot train the model.")
    exit(1)

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, [col for col in numerical_features if col in X.columns]),
        ('cat', categorical_transformer, [col for col in categorical_features if col in X.columns])
    ],
    remainder='drop'
)

# Create the model
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Create and fit the pipeline
print("Training the model...")
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Model performance:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Save the model
print("Saving the model...")
model_directory = "models"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model_path = os.path.join(model_directory, "churn_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)

print(f"Model saved to {model_path}")
print("Done!")