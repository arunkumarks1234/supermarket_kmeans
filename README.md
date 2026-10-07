# Supermarket Segmentation (Full Stack)

This project performs K-Means clustering on supermarket sales data to segment shoppers based on purchasing behavior. 

## Structure
- `backend/`: FastAPI application containing ML logic for K-Means and EDA.
- `frontend/`: React + Vite dashboard displaying the ML results.
- `data/`: Place your `SuperMarket Analysis.csv` here.
- `output/`: Generated CSVs, models, and plot images.

## Setup
### Local (Windows)
Run `.\setup.ps1`

### Local (Mac/Linux)
Run `./setup.sh`

## Running
**Backend**:
```bash
cd backend
# Windows:
.\.venv\Scripts\uvicorn.exe app.main:app --reload
# Mac/Linux:
source .venv/bin/activate
uvicorn app.main:app --reload
```

**Frontend**:
```bash
cd frontend
npm run dev
```

## Docker Compose
You can also run everything via Docker:
```bash
docker-compose up --build
```
