"""
Script to create an admin user in the local database
This script creates an admin user with specified credentials
"""
import os
import sys
from werkzeug.security import generate_password_hash
from app import app, db
from models import User

def create_admin_user(email, password, first_name="Admin", last_name="User"):
    """
    Create an admin user with the specified credentials
    
    Args:
        email (str): Admin email
        password (str): Admin password
        first_name (str): Admin first name
        last_name (str): Admin last name
    """
    try:
        with app.app_context():
            # Check if user already exists
            existing_user = User.query.filter_by(email=email).first()
            
            if existing_user:
                print(f"User {email} already exists.")
                
                # Update password if requested
                update = input("Update password? (y/n): ").lower().strip() == 'y'
                if update:
                    existing_user.password_hash = generate_password_hash(password)
                    existing_user.is_admin = True  # Ensure admin status
                    db.session.commit()
                    print(f"Password updated for {email}")
                
                # Update admin status if not already admin
                if not existing_user.is_admin:
                    make_admin = input("Make this user an admin? (y/n): ").lower().strip() == 'y'
                    if make_admin:
                        existing_user.is_admin = True
                        db.session.commit()
                        print(f"User {email} is now an admin")
                
                return
            
            # Create new admin user
            admin_user = User(
                email=email,
                password_hash=generate_password_hash(password),
                first_name=first_name,
                last_name=last_name,
                is_admin=True,
                active_status=True,
                login_count=0
            )
            
            db.session.add(admin_user)
            db.session.commit()
            print(f"Admin user {email} created successfully!")
    
    except Exception as e:
        print(f"Error creating admin user: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("========== NetOne Admin User Creation ==========")
    
    # Set default values
    default_email = "admin@gmail.com"
    default_password = "Adminpassword@123"
    default_first_name = "Admin"
    default_last_name = "User"
    
    # Get inputs with defaults
    email = input(f"Admin email [default: {default_email}]: ") or default_email
    password = input(f"Admin password [default: {default_password}]: ") or default_password
    first_name = input(f"First name [default: {default_first_name}]: ") or default_first_name
    last_name = input(f"Last name [default: {default_last_name}]: ") or default_last_name
    
    # Create the admin user
    success = create_admin_user(email, password, first_name, last_name)
    
    if success:
        print(f"\nAdmin user ready to use!")
        print(f"Login with: {email} / {password}")
    else:
        print("\nFailed to create admin user. Please check database connection and try again.")