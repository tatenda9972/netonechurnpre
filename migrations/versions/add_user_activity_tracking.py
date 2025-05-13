"""add user activity tracking

Revision ID: add_user_activity_tracking
Revises: 
Create Date: 2025-05-13 12:04:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_user_activity_tracking'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add active_status, last_login, and login_count columns to user table
    op.add_column('user', sa.Column('active_status', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('user', sa.Column('last_login', sa.DateTime(), nullable=True))
    op.add_column('user', sa.Column('login_count', sa.Integer(), nullable=False, server_default='0'))


def downgrade():
    # Remove the columns if needed
    op.drop_column('user', 'login_count')
    op.drop_column('user', 'last_login')
    op.drop_column('user', 'active_status')