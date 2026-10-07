import pytest
import pandas as pd
import numpy as np
from app.ml.preprocess import get_summary, clean_data, check_outliers

def test_get_summary():
    df = pd.DataFrame({
        'A': [1, 2, np.nan],
        'B': ['x', 'y', 'z']
    })
    summary = get_summary(df)
    assert summary['shape'] == [3, 2]
    assert summary['missing']['A'] == 1

def test_clean_data():
    df = pd.DataFrame({
        'A': [1, 1, 2]
    })
    clean = clean_data(df)
    assert len(clean) == 2

def test_check_outliers():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100] # 100 is an outlier
    })
    outliers = check_outliers(df)
    assert outliers['A']['count'] > 0
