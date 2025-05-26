"""
Fix SQLite database schema by adding missing columns
This script will add the missing columns to make the app work properly
"""
import sqlite3
import os
from app import app

def fix_database():
    """Add missing columns to the SQLite database"""
    
    # Get the database path
    sqlite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'netone.db')
    
    # Make sure the instance directory exists
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    
    print(f"Fixing database at: {sqlite_path}")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        
        # Check current table structure
        cursor.execute("PRAGMA table_info(user)")
        columns = {row[1]: row for row in cursor.fetchall()}
        print(f"Current user table columns: {list(columns.keys())}")
        
        # Add missing columns if they don't exist
        if 'active_status' not in columns:
            print("Adding active_status column...")
            cursor.execute("ALTER TABLE user ADD COLUMN active_status BOOLEAN DEFAULT 1 NOT NULL")
            print("✓ active_status column added")
        
        if 'last_login' not in columns:
            print("Adding last_login column...")
            cursor.execute("ALTER TABLE user ADD COLUMN last_login TIMESTAMP")
            print("✓ last_login column added")
        
        if 'login_count' not in columns:
            print("Adding login_count column...")
            cursor.execute("ALTER TABLE user ADD COLUMN login_count INTEGER DEFAULT 0 NOT NULL")
            print("✓ login_count column added")
        
        # Commit the changes
        conn.commit()
        
        # Verify the columns were added
        cursor.execute("PRAGMA table_info(user)")
        updated_columns = {row[1]: row for row in cursor.fetchall()}
        print(f"Updated user table columns: {list(updated_columns.keys())}")
        
        # Close the connection
        conn.close()
        
        print("✅ Database schema fixed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing database: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("NetOne Database Schema Fix")
    print("=" * 50)
    
    success = fix_database()
    
    if success:
        print("\n🎉 Your database is now ready!")
        print("You can start the application and login/register should work.")
    else:
        print("\n❌ Database fix failed. Please check the error messages above.")