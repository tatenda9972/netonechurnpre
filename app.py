import os
import datetime
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_wtf.csrf import CSRFProtect
from utils import get_retention_recommendations

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "netone_secret_key_development")

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Set up proxy middleware
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
database_url = os.environ.get("DATABASE_URL")

# Update the URL for SQLAlchemy if using postgres://
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

# Default to SQLite for local development
sqlite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'netone.db')
app.config["SQLALCHEMY_DATABASE_URI"] = database_url or f"sqlite:///{sqlite_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Make sure instance folder exists
os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)

print(f"Using database: {app.config['SQLALCHEMY_DATABASE_URI']}")

# Initialize the database
db.init_app(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"

# Initialize the database tables with app context
with app.app_context():
    # Import models here to avoid circular imports
    from models import User, Prediction
    
    try:
        # Try to create all tables
        db.create_all()
        print("Database tables created or already exist")
    except Exception as e:
        print(f"Note: Error creating tables - they may already exist: {str(e)}")

    # Add load_user callback for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        try:
            return User.query.get(int(user_id))
        except Exception as e:
            print(f"Error loading user: {str(e)}")
            # If there's a database error (like missing columns), 
            # return None to prevent app crash
            return None

# Configure Flask app to use current datetime in templates and add helper functions
@app.context_processor
def inject_now():
    return {
        'now': datetime.datetime.now(),
        'get_retention_recommendations': get_retention_recommendations
    }
