"""Initial database migration

Revision ID: 1a2b3c4d5e6f
Create Date: 2025-08-12 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers
revision = '1a2b3c4d5e6f'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create portfolios table
    op.create_table('portfolios',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('mode', sa.String(), nullable=False),
        sa.Column('cash', sa.Float(), nullable=False),
        sa.Column('starting_balance', sa.Float(), nullable=False),
        sa.Column('realized_pnl', sa.Float(), nullable=False),
        sa.Column('unrealized_pnl', sa.Float(), nullable=False),
        sa.Column('last_updated', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_portfolios_mode', 'portfolios', ['mode'])

    # Create holdings table
    op.create_table('holdings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portfolio_id', sa.Integer(), nullable=False),
        sa.Column('ticker', sa.String(), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('avg_price', sa.Float(), nullable=False),
        sa.Column('last_price', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_holdings_ticker', 'holdings', ['ticker'])
    op.create_index('idx_holdings_portfolio', 'holdings', ['portfolio_id'])

    # Create trades table
    op.create_table('trades',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portfolio_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('ticker', sa.String(), nullable=False),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('pnl', sa.Float(), nullable=True),
        sa.Column('stop_loss', sa.Float(), nullable=True),
        sa.Column('take_profit', sa.Float(), nullable=True),
        sa.Column('metadata', sqlite.JSON, nullable=True),
        sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_trades_timestamp', 'trades', ['timestamp'])
    op.create_index('idx_trades_ticker', 'trades', ['ticker'])
    op.create_index('idx_trades_portfolio', 'trades', ['portfolio_id'])

def downgrade():
    op.drop_table('trades')
    op.drop_table('holdings')
    op.drop_table('portfolios')
