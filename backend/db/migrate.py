"""
Migration script to transfer data from JSON files to SQLite database
"""
import os
import sys
from pathlib import Path

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from db.database import DatabaseManager

def main():
    """Run the migration"""
    # Get project root directory
    project_dir = Path(__file__).parent.parent.parent
    data_dir = os.path.join(project_dir, 'data')
    db_path = f'sqlite:///{os.path.join(data_dir, "trading.db")}'
    
    print(f"Starting migration from {data_dir} to {db_path}")
    
    # Initialize database manager
    db = DatabaseManager(db_path)
    
    try:
        # Run migration
        db.migrate_json_to_sqlite(data_dir)
        print("Migration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        raise

if __name__ == "__main__":
    main()
