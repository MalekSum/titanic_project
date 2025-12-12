# src/utils.py

import numpy as np

def cap_outliers_iqr(series):
    """Cap outliers using IQR."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return np.where(series > upper, upper,
                    np.where(series < lower, lower, series))
