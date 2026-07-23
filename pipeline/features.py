"""
feature construction
"""

import pandas as pd
import numpy as np
 
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158


#### Alpha158 feature construction via Qlib

provider_uri="~/.qlib/qlib_data/my_custom_data"
start_time="2018-01-02"
end_time="2026-06-05"
fit_start_time="2018-01-02"
fit_end_time="2023-12-29"
instruments=['000016.SH', '000300.SH', '000852.SH', '000905.SH', '000985.CSI',
       '399303.SZ', '868008.WI', '8841425.WI', '932000.CSI']
# instruments = df['code'].unique()

def build_alpha158(
    provider_uri: str,
    instruments: list[str] | str,
    start_time: str,
    end_time: str,
    fit_start_time: str,
    fit_end_time: str,
) -> pd.DataFrame:
    """
    Initialize Qlib against the given data directory and fetch Alpha158
    """
    qlib.init(provider_uri=provider_uri, region=REG_CN)
 
    handler = Alpha158(
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        instruments=instruments,
    )
 
    return handler.fetch()

def split_X_y_alpah158(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split an Alpha158 output DataFrame into X (158 feature columns) and
    y (label column), dropping any row where a feature or the label is NaN.
    """
    X = data.iloc[:, 0:158]
    y = data.iloc[:, 158]
 
    keep = X.notna().all(axis=1) & y.notna()
    X, y = X[keep], y[keep]
 
    return X, y


#### mannual feature consturction for LSTM

# helper function for features
def resi30(close_series, window=30):
    resid = pd.Series(index=close_series.index, dtype=float)
    values = close_series.values
    for i in range(window - 1, len(values)):
        y = values[i - window + 1: i + 1]
        x = np.arange(window)
        # fit linear trend: y = a*x + b
        a, b = np.polyfit(x, y, 1)
        fitted_today = a * (window - 1) + b
        resid.iloc[i] = (values[i] - fitted_today) / values[i]
    return resid

def imin20(close_series, window=20):
    result = pd.Series(index=close_series.index, dtype=float)
    values = close_series.values
    for i in range(window - 1, len(values)):
        window_vals = values[i - window + 1: i + 1]
        min_pos_from_start = np.argmin(window_vals)          
        days_ago = (window - 1) - min_pos_from_start         
        result.iloc[i] = days_ago / window
    return result


# construct the data set for LSTM
def build_df_for_LSTM(df: pd.DataFrame) -> pd.DataFrame:
    """
    construct the data frame that LSTM requires 
    can drop some unecessary columns if needed but here we did not drop
    """
    raw_df = df.copy()

    # build features
    raw_df["log_ret"] = raw_df.groupby('code')['CLOSE'].transform(lambda x: np.log(x / x.shift(1)))
    raw_df["vol_z"] = raw_df.groupby('code')["VOLUME"].transform(lambda x: (x-x.rolling(20).mean()) / x.rolling(20).std())
    raw_df["hl_range"] = (raw_df["HIGH"] - raw_df["LOW"]) / raw_df["CLOSE"]
    raw_df['STD20'] = raw_df.groupby('code')['CLOSE'].transform(lambda x: x.rolling(20).std() / x)
    raw_df['RESI30'] = raw_df.groupby('code')['CLOSE'].transform(lambda x: resi30(x))
    raw_df['IMIN20'] = raw_df.groupby('code')['CLOSE'].transform(lambda x: imin20(x))
    
    # build label
    raw_df["label"] = raw_df.groupby('code')['CLOSE'].transform(lambda x: x.shift(-2) / x.shift(-1) - 1)

    # drop null
    raw_df = raw_df.dropna()

    return raw_df

feature_cols = ["log_ret", "vol_z", "hl_range", "STD20", "RESI30", "IMIN20"]

def split_X_y_LSTM(raw_df: pd.DataFrame, feature_cols: list):
    """
    split X and y for lstm and convert it into the DataFrame structure Qlib designs for Alpha158
    to ensure data splitting function can be used consistently
    """
    X_lstm = raw_df.set_index(['date', 'code'])[feature_cols]
    X_lstm.index.names = ['datetime', 'instrument']

    y_lstm = raw_df.set_index(['date', 'code'])['label']
    y_lstm.index.names = ['datetime', 'instrument']

    return X_lstm, y_lstm


