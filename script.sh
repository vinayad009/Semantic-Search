#!/bin/bash

# Define project name
PROJECT_NAME="mindtickle_search"

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install necessary packages
echo "Installing dependencies..."
pip install -r requirements.txt

# Create a directory for uploaded files
echo "Creating uploads directory..."
mkdir -p uploads

# Run the FastAPI application
echo "Running FastAPI application..."
uvicorn app.main:app --reload

# Deactivate the virtual environment
deactivate
