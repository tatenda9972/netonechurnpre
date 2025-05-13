import os
import logging
import pandas as pd
from app import app
from routes import configure_routes
from admin_routes import configure_admin_routes
from ml_model import ChurnPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure main routes
configure_routes(app)

# Configure admin routes
configure_admin_routes(app)

def evaluate_model_on_startup():
    """
    Evaluate the ML model and display its metrics in the terminal
    This provides a quick way to ensure the model is performing as expected
    """
    print("\n" + "="*80)
    print("=" + " "*30 + "MODEL EVALUATION" + " "*30 + "=")
    print("="*80)
    
    try:
        # Load test data from the CSV file
        csv_path = 'attached_assets/netone_customers.csv'
        print(f"Loading test data from: {csv_path}")
        
        # Load only if the file exists
        if not os.path.exists(csv_path):
            print(f"Error: Test data file '{csv_path}' not found.")
            print("="*80 + "\n")
            return
        
        # Load the data
        test_data = pd.read_csv(csv_path)
        print(f"Test data loaded successfully: {len(test_data)} records with {len(test_data.columns)} features")
        
        # Get target distribution
        if 'Churn' in test_data.columns:
            churn_count = test_data['Churn'].sum()
            churn_rate = (churn_count / len(test_data)) * 100
            print(f"Churn distribution: {churn_count} churned customers ({churn_rate:.2f}%)")
        
        # Initialize the model
        print("\nLoading and initializing the churn prediction model...")
        model_path = os.path.join('models', 'churn_model.pkl')
        print(f"Model path: {model_path}")
        predictor = ChurnPredictor()
        
        # Verify if the model is properly loaded
        if predictor.pipeline is None:
            print("Error: Failed to load the model.")
            print("="*80 + "\n")
            return
        
        # Check if test data has the target column for evaluation
        if 'Churn' not in test_data.columns:
            print("Warning: 'Churn' column not found in test data. Cannot evaluate model metrics.")
            print("="*80 + "\n")
            return
        
        # Perform prediction and evaluation
        print("\nEvaluating model on test data...")
        results = predictor.predict(test_data)
        
        # Extract and display metrics
        metrics = results['metrics']
        
        print("\n" + "-"*50)
        print("MODEL PERFORMANCE METRICS".center(50))
        print("-"*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}  (Correct predictions / Total)")
        print(f"Precision: {metrics['precision']:.4f}  (True positives / Predicted positives)")
        print(f"Recall:    {metrics['recall']:.4f}  (True positives / Actual positives)")
        print(f"F1 Score:  {metrics['f1']:.4f}  (Harmonic mean of precision and recall)")
        
        # Calculate additional metrics
        cm = results['confusion_matrix']
        total = cm.sum()
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print(f"Specificity: {specificity:.4f}  (True negatives / Actual negatives)")
        print(f"NPV:       {npv:.4f}  (True negatives / Predicted negatives)")
        
        # Display confusion matrix with percentages
        print("\n" + "-"*50)
        print("CONFUSION MATRIX".center(50))
        print("-"*50)
        print(f"                   | Predicted Negative | Predicted Positive |")
        print(f"Actual Negative   | {tn:10d} ({tn/total*100:5.1f}%) | {fp:10d} ({fp/total*100:5.1f}%) |")
        print(f"Actual Positive   | {fn:10d} ({fn/total*100:5.1f}%) | {tp:10d} ({tp/total*100:5.1f}%) |")
        
        # Calculate class metrics
        class0_accuracy = (tn + tp) / total
        class0_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        class0_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        class0_f1 = 2 * (class0_precision * class0_recall) / (class0_precision + class0_recall) if (class0_precision + class0_recall) > 0 else 0
        
        # Display class metrics
        print("\n" + "-"*50)
        print("CLASS METRICS".center(50))
        print("-"*50)
        print(f"Positive class (Churn):    TP={tp}, FP={fp}, FN={fn}")
        print(f"Negative class (No Churn): TN={tn}, FP={fn}, FN={fp}")
        
        # Display feature importance
        try:
            fi = results['feature_importance']
            print("\n" + "-"*50)
            print("TOP FEATURE IMPORTANCE".center(50))
            print("-"*50)
            for i in range(min(10, len(fi['features']))):
                print(f"{i+1}. {fi['features'][i]:20s}: {fi['importance'][i]:.4f}")
                
            # Print a brief interpretation
            print("\nInterpretation: Higher values indicate features that have")
            print("more influence on the model's churn predictions.")
        except Exception as e:
            print(f"\nError retrieving feature importance: {str(e)}")
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
    
    print("\n" + "="*80)
    print("="*30 + " END OF EVALUATION " + "="*30)
    print("="*80 + "\n")

# Evaluate model on startup
evaluate_model_on_startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
