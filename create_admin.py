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
    with app.app_context():
        # Check if the user already exists
        existing_user = User.query.filter_by(email=email).first()
        
        if existing_user:
            if existing_user.is_admin:
                print(f"Admin user {email} already exists.")
                return
            else:
                # Convert existing user to admin
                existing_user.is_admin = True
                db.session.commit()
                print(f"Existing user {email} converted to admin.")
                return
        
        # Create new admin user
        admin_user = User(
            email=email,
            password_hash=generate_password_hash(password),
            first_name=first_name,
            last_name=last_name,
            is_admin=True
        )
        
        db.session.add(admin_user)
        db.session.commit()
        print(f"Admin user {email} created successfully!")

if __name__ == "__main__":
    # Default admin credentials
    admin_email = "admin@gmail.com"
    admin_password = "Adminpassword@123"
    
    # Create the admin user
    create_admin_user(admin_email, admin_password)