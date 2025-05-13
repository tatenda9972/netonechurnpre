import os
import sys
from flask import Flask
from flask.cli import FlaskGroup
from app import app, db
from models import User, Prediction

# Fix for type checking error with FlaskGroup
if 'FLASK_APP' not in os.environ:
    os.environ['FLASK_APP'] = 'app.py'

# Create CLI group
cli = FlaskGroup(create_app=lambda: app)

@cli.command("init_db")
def init_db():
    """Initialize the database."""
    db.create_all()
    print("Database initialized.")

@cli.command("db_migrate")
def db_migrate():
    """Run database migrations."""
    # Import needed modules inside the function to avoid circular imports
    from alembic.config import Config
    from alembic import command
    
    # Use the alembic.ini file
    alembic_cfg = Config("migrations/alembic.ini")
    
    # Run the migration
    command.upgrade(alembic_cfg, "head")
    print("Database migration completed.")

@cli.command("repair_db")
def repair_db():
    """Repair the database schema for missing columns (for Visual Studio Code compatibility)."""
    try:
        print(f"Repairing database schema for: {app.config['SQLALCHEMY_DATABASE_URI']}")
        
        # Run SQL to add missing columns for SQLite
        if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI']:
            from sqlite3 import connect
            db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
            
            print(f"Using SQLite path: {db_path}")
            
            with connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user table exists
                cursor.execute("PRAGMA table_info(user)")
                columns = {row[1]: row for row in cursor.fetchall()}
                
                # Add missing columns
                if 'active_status' not in columns:
                    print("Adding active_status column...")
                    cursor.execute("ALTER TABLE user ADD COLUMN active_status BOOLEAN DEFAULT 1 NOT NULL")
                
                if 'last_login' not in columns:
                    print("Adding last_login column...")
                    cursor.execute("ALTER TABLE user ADD COLUMN last_login TIMESTAMP")
                
                if 'login_count' not in columns:
                    print("Adding login_count column...")
                    cursor.execute("ALTER TABLE user ADD COLUMN login_count INTEGER DEFAULT 0 NOT NULL")
                
                conn.commit()
                
                # Verify the columns were added
                cursor.execute("PRAGMA table_info(user)")
                columns = {row[1]: row for row in cursor.fetchall()}
                print(f"Updated user columns: {list(columns.keys())}")
        
        # Run alembic migration to ensure all other tables and columns are up to date
        print("Running migrations to ensure schema is up to date...")
        from alembic.config import Config
        from alembic import command
        
        # Use the alembic.ini file
        alembic_cfg = Config("migrations/alembic.ini")
        
        # Run the migration
        command.upgrade(alembic_cfg, "head")
            
        print("✅ Database schema repair completed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error repairing database: {str(e)}")

@cli.command("verify_db")
def verify_db():
    """Verify that the database schema is compatible with the current models."""
    try:
        with app.app_context():
            # Check user table
            print("Verifying database schema compatibility...")
            
            # Get a user to validate the schema
            user_count = User.query.count()
            print(f"Found {user_count} users in the database.")
            
            # Check prediction table
            prediction_count = Prediction.query.count()
            print(f"Found {prediction_count} predictions in the database.")
            
            print("✅ Database verification completed successfully!")
    except Exception as e:
        print(f"❌ Database verification failed: {str(e)}")
        print("Run 'python manage.py repair_db' to attempt auto-repair.")

if __name__ == "__main__":
    cli()