import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def get_summary(df: pd.DataFrame) -> dict:
    """Return data summary including shape, dtypes, missing, duplicates, and describe."""
    summary = {
        "shape": list(df.shape),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "describe": df.describe().to_dict()
    }
    return summary

def check_outliers(df: pd.DataFrame) -> dict:
    """Check for outliers using IQR for numerical columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    outliers = {}
    for col in numeric_df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outliers[col] = {
            "count": count,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }
    return outliers

def get_correlation_matrix(df: pd.DataFrame) -> dict:
    """Return correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr().to_dict()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data: drop duplicates."""
    return df.drop_duplicates().copy()
