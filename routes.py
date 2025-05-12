import os
import pandas as pd
import json
from datetime import datetime
from flask import render_template, redirect, url_for, flash, request, send_file, jsonify, abort, Response
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from io import StringIO, BytesIO
import csv

from app import app, db
from models import User, Prediction
from forms import LoginForm, RegistrationForm, PredictionForm, ProfileForm
from ml_model import ChurnPredictor
from utils import validate_email, generate_report, preprocess_csv

def configure_routes(app):
    """Configure routes for the Flask app"""
    
    @app.route('/')
    def index():
        """Render the homepage"""
        return render_template('index.html', title='NetOne Churn Prediction System')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Handle user login"""
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data).first()
            
            if user and check_password_hash(user.password_hash, form.password.data):
                login_user(user, remember=form.remember_me.data)
                next_page = request.args.get('next')
                flash('Login successful!', 'success')
                return redirect(next_page or url_for('dashboard'))
            else:
                flash('Invalid email or password', 'danger')
        
        return render_template('login.html', title='Login', form=form)
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """Handle user registration"""
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        form = RegistrationForm()
        if form.validate_on_submit():
            # Check if email already exists
            if User.query.filter_by(email=form.email.data).first():
                flash('Email already registered', 'danger')
                return render_template('register.html', title='Register', form=form)
            
            # Check email format
            if not validate_email(form.email.data):
                flash('Invalid email format', 'danger')
                return render_template('register.html', title='Register', form=form)
            
            # Create new user
            hashed_password = generate_password_hash(form.password.data)
            user = User(
                email=form.email.data,
                password_hash=hashed_password,
                first_name=form.first_name.data,
                last_name=form.last_name.data
            )
            
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        
        return render_template('register.html', title='Register', form=form)
    
    @app.route('/logout')
    @login_required
    def logout():
        """Handle user logout"""
        logout_user()
        flash('You have been logged out', 'info')
        return redirect(url_for('index'))
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        """Render the user dashboard"""
        # Get recent predictions
        recent_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
            Prediction.created_at.desc()).limit(5).all()
        
        # Get total predictions count
        total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
        
        # Get average churn rate from predictions
        predictions = Prediction.query.filter_by(user_id=current_user.id).all()
        if predictions:
            avg_churn_rate = sum(p.churn_count / p.total_customers * 100 for p in predictions) / len(predictions)
        else:
            avg_churn_rate = 0
        
        return render_template(
            'dashboard.html',
            title='Dashboard',
            recent_predictions=recent_predictions,
            total_predictions=total_predictions,
            avg_churn_rate=avg_churn_rate
        )
    
    @app.route('/prediction', methods=['GET', 'POST'])
    @login_required
    def prediction():
        """Handle churn prediction from CSV upload"""
        form = PredictionForm()
        recent_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
            Prediction.created_at.desc()).limit(5).all()
        
        if form.validate_on_submit():
            try:
                # Get uploaded file
                csv_file = form.csv_file.data
                
                # Get filename and add timestamp to make it unique
                original_filename = secure_filename(csv_file.filename)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{timestamp}_{original_filename}"
                
                # Read CSV data
                try:
                    csv_data = csv_file.read().decode('utf-8')
                    df = pd.read_csv(StringIO(csv_data))
                    print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
                except Exception as e:
                    print(f"Error reading CSV file: {str(e)}")
                    flash('Error reading CSV file. Please ensure it is a valid CSV file.', 'danger')
                    return render_template('prediction.html', title='Prediction', form=form, 
                                          recent_predictions=recent_predictions)
                
                # Check if data is valid
                if len(df) == 0:
                    flash('The uploaded CSV file is empty.', 'danger')
                    return render_template('prediction.html', title='Prediction', form=form,
                                          recent_predictions=recent_predictions)
                
                # Validate required columns
                required_columns = [
                    'Age', 'Gender', 'Location', 'Tenure_Months', 
                    'Data_Usage_GB', 'Call_Minutes', 'SMS_Count'
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    missing_cols_str = ', '.join(missing_columns)
                    flash(f'Missing required columns in the CSV file: {missing_cols_str}', 'danger')
                    return render_template('prediction.html', title='Prediction', form=form,
                                          recent_predictions=recent_predictions)
                
                # Preprocess data
                try:
                    preprocessed_df = preprocess_csv(df)
                    print(f"Preprocessed data shape: {preprocessed_df.shape}")
                except Exception as e:
                    print(f"Error during data preprocessing: {str(e)}")
                    flash(f'Error preprocessing data: {str(e)}', 'danger')
                    return render_template('prediction.html', title='Prediction', form=form,
                                          recent_predictions=recent_predictions)
                
                # Make predictions
                try:
                    predictor = ChurnPredictor()
                    
                    # Check if model was loaded successfully
                    if predictor.pipeline is None:
                        print("Model not loaded successfully, using fallback")
                        flash('Warning: Using fallback prediction as the model could not be loaded.', 'warning')
                    
                    result = predictor.predict(preprocessed_df)
                    print("Prediction completed successfully")
                except Exception as e:
                    print(f"Error during prediction: {str(e)}")
                    flash(f'Error during prediction: {str(e)}', 'danger')
                    return render_template('prediction.html', title='Prediction', form=form,
                                          recent_predictions=recent_predictions)
                
                # Store prediction details with error handling
                try:
                    # Check if the prediction field exists in each row and add if missing
                    customer_df = result['customer_predictions']
                    
                    # Ensure churn_probability exists and has no nulls
                    if 'churn_probability' not in customer_df.columns:
                        print("Adding missing churn_probability column")
                        customer_df['churn_probability'] = 0.0
                    
                    # Ensure prediction column exists and has no nulls
                    if 'prediction' not in customer_df.columns:
                        print("Adding missing prediction column")
                        customer_df['prediction'] = 0
                    
                    # Convert DataFrame to dictionary in a way that preserves row data
                    # Using orient='index' makes each row a dictionary with column names as keys
                    customer_predictions_dict = customer_df.to_dict(orient='index')
                    
                    # Print sample of what's being saved for debugging
                    sample_keys = list(customer_predictions_dict.keys())[:2]
                    for key in sample_keys:
                        row_data = customer_predictions_dict[key]
                        # Check each row has required fields
                        if 'churn_probability' not in row_data:
                            print(f"Adding missing churn_probability to row {key}")
                            customer_predictions_dict[key]['churn_probability'] = 0.0
                        if 'prediction' not in row_data:
                            print(f"Adding missing prediction to row {key}")
                            customer_predictions_dict[key]['prediction'] = 0
                        
                        print(f"Sample row {key}: {customer_predictions_dict[key]}")
                    
                    prediction_details = {
                        'customer_predictions': customer_predictions_dict,
                        'confusion_matrix': result['confusion_matrix'].tolist() if 'confusion_matrix' in result else None
                    }
                except Exception as e:
                    print(f"Error creating prediction details: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    prediction_details = {
                        'customer_predictions': {},
                        'confusion_matrix': None
                    }
                
                # Store feature importance with error handling
                try:
                    feature_importance = {
                        'features': result['feature_importance']['features'],
                        'importance': result['feature_importance']['importance']
                    }
                except Exception as e:
                    print(f"Error creating feature importance: {str(e)}")
                    feature_importance = {
                        'features': ['Feature 1', 'Feature 2', 'Feature 3'],
                        'importance': [0.5, 0.3, 0.2]
                    }
                
                # Calculate churn count safely
                churn_count = 0
                try:
                    if isinstance(result['customer_predictions'], pd.DataFrame):
                        if 'prediction' in result['customer_predictions'].columns:
                            churn_count = int(result['customer_predictions']['prediction'].sum())
                    else:
                        # Dictionary format
                        prediction_col = result['customer_predictions'].get('prediction', {})
                        if prediction_col:
                            churn_count = sum(1 for val in prediction_col.values() if val == 1)
                except Exception as e:
                    print(f"Error calculating churn count: {str(e)}")
                    churn_count = 0
                
                # Get metrics with error handling
                try:
                    accuracy = result['metrics']['accuracy']
                    precision = result['metrics']['precision']
                    recall = result['metrics']['recall']
                    f1 = result['metrics']['f1']
                except Exception as e:
                    print(f"Error accessing metrics: {str(e)}")
                    accuracy = 0.0
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                
                # Convert complex objects to JSON with error handling
                try:
                    prediction_details_json = json.dumps(prediction_details)
                    feature_importance_json = json.dumps(feature_importance)
                except Exception as e:
                    print(f"Error converting to JSON: {str(e)}")
                    prediction_details_json = json.dumps({})
                    feature_importance_json = json.dumps({})
                
                # Save prediction to database
                try:
                    prediction = Prediction(
                        user_id=current_user.id,
                        file_name=original_filename,
                        total_customers=len(preprocessed_df),
                        churn_count=churn_count,
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1,
                        prediction_details=prediction_details_json,
                        feature_importance=feature_importance_json
                    )
                    
                    db.session.add(prediction)
                    db.session.commit()
                    
                    flash('Prediction completed successfully!', 'success')
                    return redirect(url_for('prediction_result', prediction_id=prediction.id))
                except Exception as e:
                    db.session.rollback()
                    print(f"Database error: {str(e)}")
                    flash(f'Error saving prediction: {str(e)}', 'danger')
                    return render_template('prediction.html', title='Prediction', form=form,
                                          recent_predictions=recent_predictions)
            
            except Exception as e:
                print(f"General error in prediction route: {str(e)}")
                flash(f'Error processing file: {str(e)}', 'danger')
                return render_template('prediction.html', title='Prediction', form=form,
                                       recent_predictions=recent_predictions)
        
        return render_template('prediction.html', title='Prediction', form=form, 
                              recent_predictions=recent_predictions)
    
    @app.route('/prediction/<int:prediction_id>')
    @login_required
    def prediction_result(prediction_id):
        """Display prediction results"""
        try:
            prediction = Prediction.query.get_or_404(prediction_id)
            
            # Check if user owns this prediction
            if prediction.user_id != current_user.id and not current_user.is_admin:
                abort(403)
            
            # Get prediction details and feature importance
            details = prediction.get_details()
            feature_importance = prediction.get_feature_importance()
            
            # Add detailed debug logging
            print(f"Prediction details structure: {list(details.keys())}")
            
            # Make sure customer_predictions exists and is a dictionary
            if 'customer_predictions' not in details:
                print("No customer_predictions found in details - adding empty dict")
                details['customer_predictions'] = {}
            elif not isinstance(details['customer_predictions'], dict):
                print(f"customer_predictions is not a dict: {type(details['customer_predictions'])}")
                details['customer_predictions'] = {}
                
            # Get a sample of the data for debugging
            if details['customer_predictions']:
                sample_keys = list(details['customer_predictions'].keys())[:2]
                for key in sample_keys:
                    print(f"Sample row {key}: {details['customer_predictions'][key]}")
                    # Ensure each row has required fields
                    row_data = details['customer_predictions'][key]
                    if isinstance(row_data, dict):
                        # Add missing fields with defaults if necessary
                        if 'prediction' not in row_data:
                            row_data['prediction'] = 0
                        if 'churn_probability' not in row_data:
                            row_data['churn_probability'] = 0.0
            
            # Generate recent predictions list for the form view
            recent_predictions = Prediction.query.filter_by(user_id=current_user.id) \
                .order_by(Prediction.created_at.desc()).limit(5).all()
            
            return render_template(
                'prediction.html',
                title='Prediction Results',
                prediction=prediction,
                details=details,
                feature_importance=feature_importance,
                form=PredictionForm(),
                recent_predictions=recent_predictions
            )
        except Exception as e:
            print(f"Error in prediction_result: {str(e)}")
            # Add more detailed error logging
            import traceback
            traceback.print_exc()
            # Create a user-friendly error message
            flash("An error occurred while trying to view the prediction details. The issue has been logged.", "danger")
            return redirect(url_for('history'))
    
    @app.route('/history')
    @login_required
    def history():
        """Display prediction history"""
        page = request.args.get('page', 1, type=int)
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
            Prediction.created_at.desc()).paginate(page=page, per_page=10)
        
        # Debug information
        print(f"History route: Found {len(predictions.items) if predictions.items else 0} predictions for user {current_user.id}")
        
        # Make sure we have some data for the churn timeline chart
        # by retrieving a slightly larger dataset for charts
        chart_data_query = Prediction.query.filter_by(user_id=current_user.id).order_by(
            Prediction.created_at.desc()).limit(20).all()
        
        print(f"Retrieved {len(chart_data_query)} predictions for chart data")
        
        return render_template(
            'history.html',
            title='Prediction History',
            predictions=predictions
        )
    
    @app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
    @login_required
    def delete_prediction(prediction_id):
        """Delete a prediction"""
        try:
            prediction = Prediction.query.get_or_404(prediction_id)
            
            # Check if user owns this prediction or is admin
            if prediction.user_id != current_user.id and not current_user.is_admin:
                abort(403)
            
            db.session.delete(prediction)
            db.session.commit()
            
            flash('Prediction deleted successfully', 'success')
        except Exception as e:
            db.session.rollback()
            print(f"Error deleting prediction: {str(e)}")
            flash(f'Error deleting prediction: {str(e)}', 'danger')
        
        return redirect(url_for('history'))
    
    @app.route('/download_report/<int:prediction_id>')
    @login_required
    def download_report(prediction_id):
        """Download prediction results as CSV"""
        prediction = Prediction.query.get_or_404(prediction_id)
        
        # Check if user owns this prediction
        if prediction.user_id != current_user.id and not current_user.is_admin:
            abort(403)
        
        # Generate CSV report
        csv_content = generate_report(prediction)
        
        # Create response
        output = BytesIO()
        output.write(csv_content.encode('utf-8'))
        output.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"churn_prediction_{timestamp}.csv"
        
        return send_file(
            output,
            as_attachment=True,
            mimetype='text/csv',
            download_name=filename
        )
    
    @app.route('/insights')
    @login_required
    def insights():
        """Show insights page with visualizations"""
        # Get all user predictions
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
            Prediction.created_at.desc()).all()
        
        # Prepare data for charts
        timeline_data = {
            'dates': [p.created_at.strftime("%Y-%m-%d") for p in predictions],
            'churn_rates': [p.churn_count / p.total_customers * 100 for p in predictions]
        }
        
        # Get most recent prediction for detailed analysis
        recent_prediction = None
        if predictions:
            recent_prediction = predictions[0]
        
        return render_template(
            'insights.html',
            title='Insights',
            predictions=predictions,
            timeline_data=timeline_data,
            recent_prediction=recent_prediction
        )
    
    @app.route('/profile', methods=['GET', 'POST'])
    @login_required
    def profile():
        """User profile management"""
        form = ProfileForm()
        
        if request.method == 'GET':
            form.first_name.data = current_user.first_name
            form.last_name.data = current_user.last_name
        
        if form.validate_on_submit():
            # Verify current password
            if not check_password_hash(current_user.password_hash, form.current_password.data):
                flash('Current password is incorrect', 'danger')
                return render_template('profile.html', title='Profile', form=form)
            
            # Update user details
            current_user.first_name = form.first_name.data
            current_user.last_name = form.last_name.data
            
            # Update password if new password provided
            if form.new_password.data:
                current_user.password_hash = generate_password_hash(form.new_password.data)
            
            db.session.commit()
            flash('Profile updated successfully', 'success')
            return redirect(url_for('profile'))
        
        return render_template('profile.html', title='Profile', form=form)
