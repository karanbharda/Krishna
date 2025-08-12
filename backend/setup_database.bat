@echo off
echo Installing required packages...
pip install -r requirements.txt

echo Creating database and migrating data...
python initialize_db.py

echo Running database migrations...
alembic upgrade head

echo Setup complete!
pause
