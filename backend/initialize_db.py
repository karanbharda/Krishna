#!/usr/bin/env python3
"""
Initialize SQLite database and migrate existing JSON data
"""

import os
import sys
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.db.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Get data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        # Initialize database manager
        db_path = f'sqlite:///{os.path.join(data_dir, "trading.db")}'
        db_manager = DatabaseManager(db_path)
        
        # Migrate data
        logger.info("Starting data migration...")
        db_manager.migrate_json_to_sqlite(data_dir)
        logger.info("Data migration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
