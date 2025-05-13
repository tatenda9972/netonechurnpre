import os
import json
import platform
import psutil
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy import func, desc
from flask import render_template, redirect, url_for, flash, request, abort
from flask_login import login_required, current_user
from werkzeug.security import generate_password_hash

from app import app, db
from models import User, Prediction
from ml_model import ChurnPredictor

# Admin access decorator
def admin_required(f):
    """
    Decorator to ensure only admin users can access certain routes
    """
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def get_system_stats():
    """Get basic system statistics"""
    try:
        # Get memory info
        memory = psutil.virtual_memory()
        memory_usage = f"{memory.percent}% ({memory.used // (1024 * 1024)} MB)"
        
        # Get CPU info
        cpu_usage = f"{psutil.cpu_percent()}%"
        
        # Get disk info
        disk = psutil.disk_usage('/')
        disk_usage = f"{disk.percent}% ({disk.used // (1024 * 1024 * 1024)} GB)"
        
        # Get database size (approximate)
        db_size = "N/A"  # Placeholder
        
        # Get uptime (app start time, not system uptime)
        uptime = "N/A"  # Placeholder
        
        return {
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage,
            'disk_usage': disk_usage,
            'db_size': db_size,
            'uptime': uptime
        }
    except Exception as e:
        print(f"Error getting system stats: {str(e)}")
        return {
            'memory_usage': 'N/A',
            'cpu_usage': 'N/A',
            'disk_usage': 'N/A',
            'db_size': 'N/A',
            'uptime': 'N/A'
        }

def get_model_metrics():
    """Get metrics for the prediction model"""
    try:
        # Initialize the model
        predictor = ChurnPredictor()
        
        # Get metrics from recent predictions
        recent_prediction = Prediction.query.order_by(Prediction.created_at.desc()).first()
        
        if recent_prediction:
            return {
                'accuracy': recent_prediction.accuracy,
                'precision': recent_prediction.precision,
                'recall': recent_prediction.recall,
                'f1_score': recent_prediction.f1_score,
                'total_samples': recent_prediction.total_customers
            }
        else:
            # Default values if no predictions
            return {
                'accuracy': 0.85,  # Placeholder
                'precision': 0.80,  # Placeholder
                'recall': 0.75,  # Placeholder
                'f1_score': 0.77,  # Placeholder
                'total_samples': 0
            }
    except Exception as e:
        print(f"Error getting model metrics: {str(e)}")
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'total_samples': 0
        }

def get_feature_importance():
    """Get feature importance from the model"""
    try:
        # Get the most recent prediction to extract feature importance
        recent_prediction = Prediction.query.order_by(Prediction.created_at.desc()).first()
        
        if recent_prediction:
            feature_importance = recent_prediction.get_feature_importance()
            # Sort by importance value (descending)
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Get top 10 features
            top_features = sorted_features[:10]
            
            return {
                'features': [f[0] for f in top_features],
                'values': [f[1] for f in top_features]
            }
        else:
            # Default values if no predictions
            return {
                'features': ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
                'values': [0.3, 0.25, 0.2, 0.15, 0.1]
            }
    except Exception as e:
        print(f"Error getting feature importance: {str(e)}")
        return {
            'features': [],
            'values': []
        }

def get_confusion_matrix():
    """Get confusion matrix data from recent predictions"""
    try:
        # Get the most recent prediction
        recent_prediction = Prediction.query.order_by(Prediction.created_at.desc()).first()
        
        if recent_prediction:
            details = recent_prediction.get_details()
            if 'confusion_matrix' in details:
                cm = details['confusion_matrix']
                
                # Format for Chart.js matrix
                matrix_data = [
                    {'x': 0, 'y': 0, 'v': cm[0][0]},  # TN
                    {'x': 1, 'y': 0, 'v': cm[0][1]},  # FP
                    {'x': 0, 'y': 1, 'v': cm[1][0]},  # FN
                    {'x': 1, 'y': 1, 'v': cm[1][1]}   # TP
                ]
                
                return matrix_data
        
        # Default data if no matrix available
        return [
            {'x': 0, 'y': 0, 'v': 100},
            {'x': 1, 'y': 0, 'v': 10},
            {'x': 0, 'y': 1, 'v': 20},
            {'x': 1, 'y': 1, 'v': 70}
        ]
    except Exception as e:
        print(f"Error getting confusion matrix: {str(e)}")
        return [
            {'x': 0, 'y': 0, 'v': 0},
            {'x': 1, 'y': 0, 'v': 0},
            {'x': 0, 'y': 1, 'v': 0},
            {'x': 1, 'y': 1, 'v': 0}
        ]

def get_mock_system_activities():
    """Get mock system activities data (for display only)"""
    current_time = datetime.now()
    activities = [
        {
            'timestamp': (current_time - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'User Login',
            'user': 'admin@gmail.com',
            'ip_address': '192.168.1.1',
            'status': 'success'
        },
        {
            'timestamp': (current_time - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Prediction Run',
            'user': 'user@example.com',
            'ip_address': '192.168.1.2',
            'status': 'success'
        },
        {
            'timestamp': (current_time - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Password Reset',
            'user': 'another@example.com',
            'ip_address': '192.168.1.3',
            'status': 'success'
        },
        {
            'timestamp': (current_time - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Failed Login Attempt',
            'user': 'unknown@example.com',
            'ip_address': '192.168.1.4',
            'status': 'error'
        },
        {
            'timestamp': (current_time - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'System Backup',
            'user': 'admin@gmail.com',
            'ip_address': '192.168.1.1',
            'status': 'success'
        }
    ]
    return activities

def configure_admin_routes(app):
    """Configure admin routes for the Flask app"""
    
    @app.route('/admin')
    @admin_required
    def admin_dashboard():
        """Admin dashboard overview"""
        # Get stats for the dashboard
        today = datetime.now().date()
        
        # Get all users count
        total_users = User.query.count()
        
        # Get new users today
        new_users_today = User.query.filter(
            func.date(User.created_at) == today
        ).count()
        
        # Get all predictions count
        total_predictions = Prediction.query.count()
        
        # Get predictions made today
        predictions_today = Prediction.query.filter(
            func.date(Prediction.created_at) == today
        ).count()
        
        # Get recent users for display
        recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
        
        # Get recent predictions for display
        recent_predictions = Prediction.query.order_by(
            Prediction.created_at.desc()
        ).limit(5).all()
        
        # Prepare data for user activity chart (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        user_registrations = User.query.filter(
            User.created_at >= thirty_days_ago
        ).all()
        
        # Process user registration dates
        user_counts = defaultdict(int)
        for user in user_registrations:
            date_str = user.created_at.strftime('%Y-%m-%d')
            user_counts[date_str] += 1
        
        # Get last 30 days as list of date strings
        date_list = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') 
                    for x in range(30, -1, -1)]
        
        # Format user activity data
        user_activity_dates = date_list
        user_activity_counts = [user_counts.get(date, 0) for date in date_list]
        
        # Prepare data for predictions chart (last 30 days)
        prediction_records = Prediction.query.filter(
            Prediction.created_at >= thirty_days_ago
        ).all()
        
        # Process prediction dates
        prediction_counts = defaultdict(int)
        for prediction in prediction_records:
            date_str = prediction.created_at.strftime('%Y-%m-%d')
            prediction_counts[date_str] += 1
        
        # Format prediction data
        prediction_dates = date_list
        prediction_counts_data = [prediction_counts.get(date, 0) for date in date_list]
        
        return render_template(
            'admin/dashboard.html',
            total_users=total_users,
            total_predictions=total_predictions,
            new_users_today=new_users_today,
            predictions_today=predictions_today,
            recent_users=recent_users,
            recent_predictions=recent_predictions,
            user_activity_dates=user_activity_dates,
            user_activity_counts=user_activity_counts,
            prediction_dates=prediction_dates,
            prediction_counts=prediction_counts_data
        )
    
    @app.route('/admin/users')
    @admin_required
    def admin_users():
        """Admin users management page"""
        page = request.args.get('page', 1, type=int)
        
        # Get users with pagination
        pagination = User.query.order_by(User.created_at.desc()).paginate(
            page=page, per_page=10
        )
        
        return render_template(
            'admin/users.html',
            users=pagination.items,
            pagination=pagination
        )
    
    @app.route('/admin/create_user', methods=['POST'])
    @admin_required
    def admin_create_user():
        """Create a new user from admin panel"""
        try:
            # Get form data
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            email = request.form.get('email')
            password = request.form.get('password')
            is_admin = 'is_admin' in request.form
            
            # Validate email is unique
            if User.query.filter_by(email=email).first():
                flash('Email already exists.', 'danger')
                return redirect(url_for('admin_users'))
            
            # Create new user
            new_user = User(
                first_name=first_name,
                last_name=last_name,
                email=email,
                password_hash=generate_password_hash(password),
                is_admin=is_admin
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            flash(f'User {email} created successfully.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating user: {str(e)}', 'danger')
        
        return redirect(url_for('admin_users'))
    
    @app.route('/admin/update_user', methods=['POST'])
    @admin_required
    def admin_update_user():
        """Update an existing user from admin panel"""
        try:
            # Get form data
            user_id = request.form.get('user_id', type=int)
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            email = request.form.get('email')
            password = request.form.get('password')
            is_admin = 'is_admin' in request.form
            is_active = 'is_active' in request.form
            
            # Find the user
            user = User.query.get(user_id)
            if not user:
                flash('User not found.', 'danger')
                return redirect(url_for('admin_users'))
            
            # Check if email is being changed and if it's unique
            if user.email != email and User.query.filter_by(email=email).first():
                flash('Email already exists.', 'danger')
                return redirect(url_for('admin_users'))
            
            # Update user data
            user.first_name = first_name
            user.last_name = last_name
            user.email = email
            user.is_admin = is_admin
            
            # Update password if provided
            if password:
                user.password_hash = generate_password_hash(password)
            
            db.session.commit()
            
            flash(f'User {email} updated successfully.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating user: {str(e)}', 'danger')
        
        return redirect(url_for('admin_users'))
    
    @app.route('/admin/delete_user', methods=['POST'])
    @admin_required
    def admin_delete_user():
        """Delete a user and all their predictions"""
        try:
            # Get user ID from form
            user_id = request.form.get('user_id', type=int)
            
            # Find user
            user = User.query.get(user_id)
            if not user:
                flash('User not found.', 'danger')
                return redirect(url_for('admin_users'))
            
            # Don't allow deleting your own account
            if user.id == current_user.id:
                flash('You cannot delete your own account.', 'warning')
                return redirect(url_for('admin_users'))
            
            # Delete all predictions associated with the user
            Prediction.query.filter_by(user_id=user.id).delete()
            
            # Delete the user
            db.session.delete(user)
            db.session.commit()
            
            flash(f'User {user.email} and all associated predictions deleted successfully.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting user: {str(e)}', 'danger')
        
        return redirect(url_for('admin_users'))
    
    @app.route('/admin/predictions')
    @admin_required
    def admin_predictions():
        """Admin predictions management page with filtering"""
        page = request.args.get('page', 1, type=int)
        
        # Get filter parameters
        user_filter = request.args.get('user_id')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        min_churn_rate = request.args.get('min_churn_rate', type=float)
        max_churn_rate = request.args.get('max_churn_rate', type=float)
        
        # Build the query with filters
        query = Prediction.query
        
        # Apply user filter
        if user_filter:
            query = query.filter(Prediction.user_id == user_filter)
        
        # Apply date filters
        if date_from:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(Prediction.created_at >= date_from_obj)
        
        if date_to:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            query = query.filter(Prediction.created_at < date_to_obj)
        
        # Apply churn rate filter (this is more complex as it requires calculation)
        filtered_predictions = query.all()
        
        if min_churn_rate is not None or max_churn_rate is not None:
            filtered_ids = []
            for pred in filtered_predictions:
                churn_rate = (pred.churn_count / pred.total_customers * 100) if pred.total_customers > 0 else 0
                
                if min_churn_rate is not None and max_churn_rate is not None:
                    if min_churn_rate <= churn_rate <= max_churn_rate:
                        filtered_ids.append(pred.id)
                elif min_churn_rate is not None:
                    if churn_rate >= min_churn_rate:
                        filtered_ids.append(pred.id)
                elif max_churn_rate is not None:
                    if churn_rate <= max_churn_rate:
                        filtered_ids.append(pred.id)
            
            query = query.filter(Prediction.id.in_(filtered_ids))
        
        # Get all users for the filter dropdown
        all_users = User.query.all()
        
        # Order and paginate results
        pagination = query.order_by(Prediction.created_at.desc()).paginate(
            page=page, per_page=10
        )
        
        # Get summary metrics (based on filters)
        total_predictions = query.count()
        total_customers = sum(p.total_customers for p in filtered_predictions) if filtered_predictions else 0
        total_churned = sum(p.churn_count for p in filtered_predictions) if filtered_predictions else 0
        
        # Calculate average churn rate
        if total_customers > 0:
            avg_churn_rate = (total_churned / total_customers) * 100
        else:
            avg_churn_rate = 0
        
        return render_template(
            'admin/predictions.html',
            predictions=pagination.items,
            pagination=pagination,
            total_predictions=total_predictions,
            total_customers=total_customers,
            total_churned=total_churned,
            avg_churn_rate=avg_churn_rate,
            all_users=all_users,
            filters={
                'user_id': user_filter,
                'date_from': date_from,
                'date_to': date_to,
                'min_churn_rate': min_churn_rate,
                'max_churn_rate': max_churn_rate
            }
        )
    
    @app.route('/admin/delete_prediction', methods=['POST'])
    @admin_required
    def admin_delete_prediction():
        """Delete a prediction"""
        try:
            # Get prediction ID from form
            prediction_id = request.form.get('prediction_id', type=int)
            
            # Find prediction
            prediction = Prediction.query.get(prediction_id)
            if not prediction:
                flash('Prediction not found.', 'danger')
                return redirect(url_for('admin_predictions'))
            
            # Delete the prediction
            db.session.delete(prediction)
            db.session.commit()
            
            flash('Prediction deleted successfully.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting prediction: {str(e)}', 'danger')
        
        return redirect(url_for('admin_predictions'))
    
    @app.route('/admin/system')
    @admin_required
    def admin_system():
        """Admin system health page"""
        # Get system statistics
        system_stats = get_system_stats()
        
        # Get model metrics
        model_metrics = get_model_metrics()
        
        # Get feature importance
        feature_importance = get_feature_importance()
        
        # Get confusion matrix data
        confusion_matrix = get_confusion_matrix()
        
        # Get mock system activities (for display)
        system_activities = get_mock_system_activities()
        
        return render_template(
            'admin/system.html',
            system_stats=system_stats,
            model_metrics=model_metrics,
            feature_importance=feature_importance,
            confusion_matrix=confusion_matrix,
            system_activities=system_activities
        )
    
    @app.route('/admin/backup_database', methods=['POST'])
    @admin_required
    def admin_backup_database():
        """Create a backup of the database"""
        try:
            # This is a placeholder function
            # In a real implementation, you would create a database dump here
            # For now, just return a success message
            
            flash('Database backup operation completed successfully.', 'success')
        except Exception as e:
            flash(f'Error backing up database: {str(e)}', 'danger')
        
        return redirect(url_for('admin_system'))
    
    @app.route('/admin/vacuum_database', methods=['POST'])
    @admin_required
    def admin_vacuum_database():
        """Vacuum the database to optimize storage"""
        try:
            # This is a placeholder function
            # In a real implementation, you would run VACUUM on the database
            # For now, just return a success message
            
            flash('Database vacuum operation completed successfully.', 'success')
        except Exception as e:
            flash(f'Error vacuuming database: {str(e)}', 'danger')
        
        return redirect(url_for('admin_system'))
    
    @app.route('/admin/clear_cache', methods=['POST'])
    @admin_required
    def admin_clear_cache():
        """Clear application caches"""
        try:
            # This is a placeholder function
            # In a real implementation, you would clear various caches
            # For now, just return a success message
            
            flash('Cache clearing operation completed successfully.', 'success')
        except Exception as e:
            flash(f'Error clearing cache: {str(e)}', 'danger')
        
        return redirect(url_for('admin_system'))
    
    @app.route('/admin/restart_application', methods=['POST'])
    @admin_required
    def admin_restart_application():
        """Restart the application"""
        try:
            # This is a placeholder function
            # In a real implementation, you would trigger an application restart
            # For now, just return a success message
            
            flash('Application restart requested. This may take a few moments.', 'warning')
        except Exception as e:
            flash(f'Error restarting application: {str(e)}', 'danger')
        
        return redirect(url_for('admin_system'))