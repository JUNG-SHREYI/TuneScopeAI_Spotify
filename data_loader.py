"""Load the original Spotify dataset without altering its contents."""

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Data" / "dataset.csv"


def load_data(file_path=None):
    """Load and return the raw CSV dataset as-is.

    The loader intentionally does not remove columns, deduplicate rows,
    impute values, or otherwise preprocess the source data.
    """
    path = Path(file_path) if file_path is not None else DATASET_PATH
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Put your CSV at Data/dataset.csv."
        )

    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)
