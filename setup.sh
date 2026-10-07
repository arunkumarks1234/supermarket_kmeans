#!/bin/bash
echo "Setting up backend..."
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..

echo "Setting up frontend..."
cd frontend
npm install
cd ..

echo "Setup complete!"
