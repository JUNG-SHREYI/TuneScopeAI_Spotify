# TuneScope AI - Spotify Dataset Explorer

TuneScope AI is a Flask application for exploring a Spotify tracks dataset and
viewing exploratory analysis, preprocessing, classification, regression,
clustering, and model-evaluation results.

## Run locally

Use Python 3.12 or newer. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. The app expects the dataset at
`Data/dataset.csv`; that dataset is included in this repository.

## Deploy on Render

The repository includes a `render.yaml` blueprint. In Render, create a new
Blueprint Instance, connect this GitHub repository, and apply the blueprint.
Render will install `requirements.txt` and start the Flask app with Gunicorn.

XGBoost and UMAP are optional. The app uses its built-in XGBoost fallback and
reports UMAP as unavailable when those optional packages are not installed.
