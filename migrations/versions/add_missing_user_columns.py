"""add missing user columns

Revision ID: add_missing_user_columns
Revises: add_user_activity_tracking
Create Date: 2025-05-13 16:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_missing_user_columns'
down_revision = 'add_user_activity_tracking'
branch_labels = None
depends_on = None


def upgrade():
    # Add the missing columns to the user table
    # Using batch mode to handle SQLite constraints
    with op.batch_alter_table('user') as batch_op:
        # Check if active_status column exists before adding it
        try:
            batch_op.add_column(sa.Column('active_status', sa.Boolean(), nullable=False, server_default='1'))
        except:
            # Column might already exist in some environments (Replit)
            pass
            
        # Check if last_login column exists before adding it
        try:
            batch_op.add_column(sa.Column('last_login', sa.DateTime(), nullable=True))
        except:
            # Column might already exist in some environments (Replit)
            pass
            
        # Check if login_count column exists before adding it
        try:
            batch_op.add_column(sa.Column('login_count', sa.Integer(), nullable=False, server_default='0'))
        except:
            # Column might already exist in some environments (Replit)
            pass


def downgrade():
    # Remove the added columns if needed
    with op.batch_alter_table('user') as batch_op:
        try:
            batch_op.drop_column('active_status')
            batch_op.drop_column('last_login')
            batch_op.drop_column('login_count')
        except:
            pass