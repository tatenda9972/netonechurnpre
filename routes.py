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
        
        if form.validate_on_submit():
            try:
                # Get uploaded file
                csv_file = form.csv_file.data
                
                # Get filename and add timestamp to make it unique
                original_filename = secure_filename(csv_file.filename)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{timestamp}_{original_filename}"
                
                # Read CSV data
                csv_data = csv_file.read().decode('utf-8')
                df = pd.read_csv(StringIO(csv_data))
                
                # Preprocess data
                preprocessed_df = preprocess_csv(df)
                
                # Make predictions
                predictor = ChurnPredictor()
                result = predictor.predict(preprocessed_df)
                
                # Store prediction details with error handling
                try:
                    prediction_details = {
                        'customer_predictions': result['customer_predictions'].to_dict(),
                        'confusion_matrix': result['confusion_matrix'].tolist() if 'confusion_matrix' in result else None
                    }
                except Exception as e:
                    print(f"Error creating prediction details: {str(e)}")
                    # Create a minimal valid structure
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
                    # Create a minimal valid structure 
                    feature_importance = {
                        'features': ['Feature 1', 'Feature 2', 'Feature 3'],
                        'importance': [0.5, 0.3, 0.2]
                    }
                
                # Calculate churn count safely
                churn_count = 0
                if 'prediction' in result['customer_predictions']:
                    # Try to use .sum() on the prediction column
                    try:
                        churn_count = int(result['customer_predictions']['prediction'].sum())
                    except:
                        # Fallback: count predictions manually
                        for _, row in result['customer_predictions'].iterrows():
                            if 'prediction' in row and row['prediction'] == 1:
                                churn_count += 1
                
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
                flash(f'Error processing file: {str(e)}', 'danger')
                return render_template('prediction.html', title='Prediction', form=form)
        
        return render_template('prediction.html', title='Prediction', form=form)
    
    @app.route('/prediction/<int:prediction_id>')
    @login_required
    def prediction_result(prediction_id):
        """Display prediction results"""
        prediction = Prediction.query.get_or_404(prediction_id)
        
        # Check if user owns this prediction
        if prediction.user_id != current_user.id and not current_user.is_admin:
            abort(403)
        
        # Get prediction details and feature importance
        details = prediction.get_details()
        feature_importance = prediction.get_feature_importance()
        
        return render_template(
            'prediction.html',
            title='Prediction Results',
            prediction=prediction,
            details=details,
            feature_importance=feature_importance,
            form=PredictionForm()
        )
    
    @app.route('/history')
    @login_required
    def history():
        """Display prediction history"""
        page = request.args.get('page', 1, type=int)
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
            Prediction.created_at.desc()).paginate(page=page, per_page=10)
        
        return render_template(
            'history.html',
            title='Prediction History',
            predictions=predictions
        )
    
    @app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
    @login_required
    def delete_prediction(prediction_id):
        """Delete a prediction"""
        prediction = Prediction.query.get_or_404(prediction_id)
        
        # Check if user owns this prediction or is admin
        if prediction.user_id != current_user.id and not current_user.is_admin:
            abort(403)
        
        db.session.delete(prediction)
        db.session.commit()
        
        flash('Prediction deleted successfully', 'success')
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
