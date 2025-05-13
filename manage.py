from flask.cli import FlaskGroup
from app import app, db

cli = FlaskGroup(app)

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

if __name__ == "__main__":
    cli()