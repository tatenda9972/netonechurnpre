import datetime
import json
from app import db
from flask_login import UserMixin

class User(UserMixin, db.Model):
    """User model for authentication"""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(64), nullable=False)
    last_name = db.Column(db.String(64), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)
    last_login = db.Column(db.DateTime, nullable=True)
    login_count = db.Column(db.Integer, default=0)
    
    # Relationship with predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.email}>'
    
    def get_activity_stats(self):
        """Get user activity statistics"""
        return {
            'prediction_count': len(self.predictions),
            'last_prediction': self.get_last_prediction_date(),
            'avg_churn_rate': self.get_average_churn_rate(),
            'days_since_joining': self.get_days_since_joining(),
            'last_login': self.last_login
        }
    
    def get_last_prediction_date(self):
        """Get the date of the user's most recent prediction"""
        if not self.predictions:
            return None
        
        return max(p.created_at for p in self.predictions)
    
    def get_average_churn_rate(self):
        """Get the average churn rate across all user predictions"""
        if not self.predictions:
            return 0
        
        total_customers = sum(p.total_customers for p in self.predictions)
        total_churned = sum(p.churn_count for p in self.predictions)
        
        if total_customers == 0:
            return 0
            
        return (total_churned / total_customers) * 100
    
    def get_days_since_joining(self):
        """Get the number of days since the user joined"""
        if not self.created_at:
            return 0
            
        delta = datetime.datetime.now() - self.created_at
        return delta.days

class Prediction(db.Model):
    """Model for storing prediction history"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_name = db.Column(db.String(128), nullable=False)
    total_customers = db.Column(db.Integer, nullable=False)
    churn_count = db.Column(db.Integer, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)
    
    # Store prediction details as JSON
    prediction_details = db.Column(db.Text, nullable=False)
    feature_importance = db.Column(db.Text, nullable=False)
    
    def get_details(self):
        """Return prediction details as a dictionary with error handling"""
        try:
            details = json.loads(self.prediction_details)
            
            # Ensure customer_predictions is a dict and has proper structure
            if 'customer_predictions' not in details or not isinstance(details['customer_predictions'], dict):
                details['customer_predictions'] = {}
                
            # Ensure each customer prediction item has the expected structure and required fields
            for customer_id, prediction_data in details['customer_predictions'].items():
                # Check if prediction_data is a dictionary
                if not isinstance(prediction_data, dict):
                    # If not a dict, create a new default dict
                    details['customer_predictions'][customer_id] = {
                        'prediction': 0,
                        'churn_probability': 0.0,
                        'Customer_ID': f"CUST{customer_id}"
                    }
                else:
                    # Make sure required fields exist in the dict
                    if 'prediction' not in prediction_data:
                        prediction_data['prediction'] = 0
                    
                    if 'churn_probability' not in prediction_data:
                        prediction_data['churn_probability'] = 0.0
            
            return details
        except Exception as e:
            print(f"Error parsing prediction details: {str(e)}")
            return {'customer_predictions': {}, 'confusion_matrix': None}
    
    def get_feature_importance(self):
        """Return feature importance as a dictionary with error handling"""
        try:
            return json.loads(self.feature_importance)
        except Exception as e:
            print(f"Error parsing feature importance: {str(e)}")
            return {'features': ['Feature 1', 'Feature 2', 'Feature 3'], 
                   'importance': [0.5, 0.3, 0.2]}
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.file_name}>'
