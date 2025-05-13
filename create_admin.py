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
            # Check if the user already exists
            existing_user = User.query.filter_by(email=email).first()
            
            if existing_user:
                print(f"User {email} already exists.")
                
                # Always update the admin status just to be sure
                if not existing_user.is_admin:
                    existing_user.is_admin = True
                    print(f"User {email} has been made an admin.")
                
                # Reset the password if needed
                reset_password = input(f"Reset password for {email}? (y/n): ").lower().strip() == 'y'
                if reset_password:
                    existing_user.password_hash = generate_password_hash(password)
                    print(f"Password updated for {email}")
                
                # Make sure the user has all required fields
                if not hasattr(existing_user, 'active_status') or existing_user.active_status is None:
                    existing_user.active_status = True
                
                # Save all changes
                db.session.commit()
                return True
            
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
            return True
            
    except Exception as e:
        print(f"Error creating admin user: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Default admin credentials
    admin_email = "admin@gmail.com"
    admin_password = "Adminpassword@123"
    admin_first_name = "Admin"
    admin_last_name = "User"
    
    # Optionally get custom values
    use_custom = input("Use custom admin credentials? (y/n) [default: n]: ").lower().strip() == 'y'
    if use_custom:
        admin_email = input(f"Admin email [default: {admin_email}]: ") or admin_email
        admin_password = input(f"Admin password [default: {admin_password}]: ") or admin_password
        admin_first_name = input(f"First name [default: {admin_first_name}]: ") or admin_first_name
        admin_last_name = input(f"Last name [default: {admin_last_name}]: ") or admin_last_name
    
    # Create the admin user
    success = create_admin_user(admin_email, admin_password, admin_first_name, admin_last_name)
    
    if success:
        print("\nAdmin user is ready!")
        print(f"You can now log in with: {admin_email} / {admin_password}")
    else:
        print("\nFailed to create admin user. Please check the error messages above.")