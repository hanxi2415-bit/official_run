"""
data splitting
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

#### hard data split

train_dates = slice("2018-01-02", "2023-12-29")
valid_dates = slice("2024-01-02", "2025-03-14")
test_dates  = slice("2025-03-17", "2026-06-05")

def fixed_date_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_dates: tuple[str, str],
    valid_dates: tuple[str, str],
    test_dates: tuple[str, str],
):
    """
    Split X/y by fixed date boundaries. Each smthg_dates argument is a
    (start, end) tuple, e.g. ("2018-01-02", "2023-12-29").
    """
    train_slice = slice(*train_dates)
    valid_slice = slice(*valid_dates)
    test_slice = slice(*test_dates)
 
    X_train, y_train = X.loc[train_slice], y.loc[train_slice]
    X_valid, y_valid = X.loc[valid_slice], y.loc[valid_slice]
    X_test, y_test = X.loc[test_slice], y.loc[test_slice]
 
    return X_train, y_train, X_valid, y_valid, X_test, y_test


#### walk-forward split

def get_walk_forward_splits(
    dates,
    initial_train_years: int,
    valid_months: int,
    test_months: int,
    step_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    dates: sorted unique datetime index
    Returns list of (train_end, valid_end, test_end) boundary dates
    """
    dates = pd.Series(sorted(dates))
    start = dates.min() # the start time does not change
    train_end = start + pd.DateOffset(years=initial_train_years)
    
    splits = []
    while True:
        valid_end = train_end + pd.DateOffset(months=valid_months)
        test_end = valid_end + pd.DateOffset(months=test_months)
        if test_end > dates.max():
            break
        splits.append((train_end, valid_end, test_end))
        train_end = train_end + pd.DateOffset(months=step_months)  # expanding: train_start stays fixed
    return splits


def get_fold_data(
        X: pd.DataFrame, # MultiIndex (instrument, datetime)
        y: pd.Series, 
        split: tuple # (train_end, valid_end, test_end)
) -> tuple[pd.DataFrame,pd.DataFrame,
           pd.DataFrame, pd.DataFrame, 
           pd.DataFrame, pd.DataFrame]:
    dt = X.index.get_level_values('datetime')
    
    train_mask = (dt >= min(dt)) & (dt <= split[0])
    valid_mask = (dt > split[0]) & (dt <= split[1])
    test_mask  = (dt > split[1])  & (dt <= split[2])
    
    return (X[train_mask], y[train_mask],
            X[valid_mask], y[valid_mask],
            X[test_mask], y[test_mask])


#### sliding window sequence for LSTM

def build_sequences(X: pd.DataFrame, y: pd.Series, seq_len: int):
    """
    Build sliding-window sequences per instrument for sequence models (LSTM).
    Returns (X_seq, y_seq) as numpy arrays.
    """
    X_seq, y_seq, dates_seq, codes_seq = [], [], [], []
    for instrument, group in X.groupby("instrument"):
        vals = group.values
        y_vals = y.loc[group.index].values
        dates = group.index.get_level_values("datetime").values

        for i in range(seq_len, len(vals)):
            X_seq.append(vals[i - seq_len:i])
            y_seq.append(y_vals[i])
            dates_seq.append(dates[i])
            codes_seq.append(instrument)

    return np.array(X_seq), np.array(y_seq), np.array(dates_seq), np.array(codes_seq)

