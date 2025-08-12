from setuptools import setup, find_packages

setup(
    name="trading-backend",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'sqlalchemy>=2.0.0',
        'alembic>=1.12.0',
        'aiosqlite>=0.19.0',
        'langgraph>=0.0.15',
    ],
)
