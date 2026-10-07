Write-Host "Setting up backend..."
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd ..

Write-Host "Setting up frontend..."
cd frontend
npm install
cd ..

Write-Host "Setup complete!"
