import re
import pandas as pd
import json
from io import StringIO
import csv

def validate_email(email):
    """
    Validate email format
    
    Args:
        email (str): Email to validate
    
    Returns:
        bool: True if email is valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def preprocess_csv(df):
    """
    Preprocess CSV data for prediction
    
    Args:
        df (pd.DataFrame): Raw DataFrame from CSV
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # List of expected columns
    expected_columns = [
        'Customer_ID', 'First_Name', 'Surname', 'Age', 'Gender', 
        'Location', 'Tenure_Months', 'Plan_Type', 'Data_Usage_GB', 
        'Call_Minutes', 'SMS_Count', 'Payment_History_12mo', 
        'Outstanding_Balance', 'Monthly_Bill', 'Num_Complaints', 
        'Support_Tickets', 'Payment_Delays', 'Churn'
    ]
    
    # Check if required columns exist
    for col in expected_columns:
        if col not in data.columns and col != 'Churn':  # Churn column is optional for prediction
            # If important column is missing, add it with default values
            if col in ['Age', 'Tenure_Months', 'Data_Usage_GB', 'Call_Minutes', 
                      'SMS_Count', 'Payment_History_12mo', 'Outstanding_Balance', 
                      'Monthly_Bill', 'Num_Complaints', 'Support_Tickets', 'Payment_Delays']:
                data[col] = 0
            elif col in ['Gender', 'Location', 'Plan_Type']:
                data[col] = 'Unknown'
    
    # Drop unnecessary columns if they exist
    columns_to_keep = [
        'Customer_ID', 'Age', 'Gender', 'Location', 'Tenure_Months', 
        'Plan_Type', 'Data_Usage_GB', 'Call_Minutes', 'SMS_Count', 
        'Payment_History_12mo', 'Outstanding_Balance', 'Monthly_Bill', 
        'Num_Complaints', 'Support_Tickets', 'Payment_Delays'
    ]
    
    if 'Churn' in data.columns:
        columns_to_keep.append('Churn')
    
    # Keep only required columns
    data = data[columns_to_keep]
    
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Gender'].fillna('Unknown', inplace=True)
    data['Location'].fillna('Unknown', inplace=True)
    data['Tenure_Months'].fillna(0, inplace=True)
    data['Plan_Type'].fillna('Unknown', inplace=True)
    data['Data_Usage_GB'].fillna(0, inplace=True)
    data['Call_Minutes'].fillna(0, inplace=True)
    data['SMS_Count'].fillna(0, inplace=True)
    data['Payment_History_12mo'].fillna(0, inplace=True)
    data['Outstanding_Balance'].fillna(0, inplace=True)
    data['Monthly_Bill'].fillna(0, inplace=True)
    data['Num_Complaints'].fillna(0, inplace=True)
    data['Support_Tickets'].fillna(0, inplace=True)
    data['Payment_Delays'].fillna(0, inplace=True)
    
    # Convert data types
    data['Age'] = data['Age'].astype(float)
    data['Tenure_Months'] = data['Tenure_Months'].astype(float)
    data['Data_Usage_GB'] = data['Data_Usage_GB'].astype(float)
    data['Call_Minutes'] = data['Call_Minutes'].astype(float)
    data['SMS_Count'] = data['SMS_Count'].astype(float)
    data['Payment_History_12mo'] = data['Payment_History_12mo'].astype(float)
    data['Outstanding_Balance'] = data['Outstanding_Balance'].astype(float)
    data['Monthly_Bill'] = data['Monthly_Bill'].astype(float)
    data['Num_Complaints'] = data['Num_Complaints'].astype(float)
    data['Support_Tickets'] = data['Support_Tickets'].astype(float)
    data['Payment_Delays'] = data['Payment_Delays'].astype(float)
    
    if 'Churn' in data.columns:
        data['Churn'] = data['Churn'].astype(int)
    
    return data

def generate_report(prediction):
    """
    Generate a CSV report from prediction results
    
    Args:
        prediction (Prediction): Prediction model object
    
    Returns:
        str: CSV content as a string
    """
    details = prediction.get_details()
    customer_predictions = pd.DataFrame(details['customer_predictions'])
    
    # Create a StringIO object to write CSV data
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header rows with metadata
    writer.writerow(['NetOne Churn Prediction Report'])
    writer.writerow(['Generated on', prediction.created_at.strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow(['File', prediction.file_name])
    writer.writerow(['Total Customers', prediction.total_customers])
    writer.writerow(['Churned Customers', prediction.churn_count])
    writer.writerow(['Churn Rate', f"{prediction.churn_count / prediction.total_customers * 100:.2f}%"])
    writer.writerow([])
    
    # Write model metrics
    writer.writerow(['Model Metrics'])
    writer.writerow(['Accuracy', f"{prediction.accuracy:.4f}"])
    writer.writerow(['Precision', f"{prediction.precision:.4f}"])
    writer.writerow(['Recall', f"{prediction.recall:.4f}"])
    writer.writerow(['F1 Score', f"{prediction.f1_score:.4f}"])
    writer.writerow([])
    
    # Write feature importance
    feature_importance = prediction.get_feature_importance()
    writer.writerow(['Feature Importance'])
    for feature, importance in zip(feature_importance['features'], feature_importance['importance']):
        writer.writerow([feature, f"{importance:.4f}"])
    writer.writerow([])
    
    # Write customer predictions
    writer.writerow(['Customer Predictions'])
    # Write headers
    headers = customer_predictions.columns.tolist()
    writer.writerow(headers)
    
    # Write data rows
    for _, row in customer_predictions.iterrows():
        writer.writerow(row.tolist())
    
    return output.getvalue()

def get_retention_recommendations(prediction):
    """
    Get retention recommendations based on prediction results
    
    Args:
        prediction (Prediction): Prediction model object
    
    Returns:
        list: List of recommendation dictionaries
    """
    details = prediction.get_details()
    df = pd.DataFrame(details['customer_predictions'])
    
    recommendations = []
    
    # High churn probability customers
    high_risk = df[df['churn_probability'] > 0.75]
    if len(high_risk) > 0:
        recommendations.append({
            'title': 'High Risk Customers',
            'description': f'There are {len(high_risk)} customers with >75% probability of churning.',
            'action': 'Immediate personal contact, offer special discounts or incentives.'
        })
    
    # Customers with complaints
    if 'Num_Complaints' in df.columns:
        complaint_customers = df[(df['prediction'] == 1) & (df['Num_Complaints'] > 0)]
        if len(complaint_customers) > 0:
            recommendations.append({
                'title': 'Address Customer Complaints',
                'description': f'{len(complaint_customers)} customers with complaints are predicted to churn.',
                'action': 'Review and resolve complaints, offer compensation or special attention.'
            })
    
    # Long-term customers at risk
    if 'Tenure_Months' in df.columns:
        loyal_customers = df[(df['prediction'] == 1) & (df['Tenure_Months'] > 24)]
        if len(loyal_customers) > 0:
            recommendations.append({
                'title': 'Loyal Customers at Risk',
                'description': f'{len(loyal_customers)} customers with >2 years tenure are predicted to churn.',
                'action': 'Implement a loyalty program with exclusive benefits and personalized offers.'
            })
    
    # High-value customers
    if 'Monthly_Bill' in df.columns:
        avg_bill = df['Monthly_Bill'].mean()
        high_value = df[(df['prediction'] == 1) & (df['Monthly_Bill'] > avg_bill)]
        if len(high_value) > 0:
            recommendations.append({
                'title': 'High-Value Customers',
                'description': f'{len(high_value)} high-value customers are predicted to churn.',
                'action': 'Provide premium services or upgrades at no additional cost.'
            })
    
    # General recommendation
    churn_count = df[df['prediction'] == 1].shape[0]
    churn_rate = churn_count / len(df) * 100
    recommendations.append({
        'title': 'Overall Churn Rate',
        'description': f'The predicted churn rate is {churn_rate:.2f}%.',
        'action': 'Review pricing strategy and service quality to improve customer satisfaction.'
    })
    
    return recommendations
