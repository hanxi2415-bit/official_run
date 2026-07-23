"""
Data reading and cleaning (unit adjustment) for the multi-stock pipeline.
"""

import os
from pathlib import Path
 
import mysql.connector
import pandas as pd
from dotenv import load_dotenv
 
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158


#### data reading
def load_raw_data(cache: str = "bench_basic_data.parquet") -> pd.DataFrame:
    """
    Load raw stock data
    Requires DB_HOST, DB_USER, DB_PASSWORD, DB_NAME in a local .env file
    use a local parquet cache after loading from API
    """
    load_dotenv()
 
    if Path(cache).exists():
        df = pd.read_parquet(cache)
    else:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )
        df = pd.read_sql("SELECT * FROM bench_basic_data", conn)
        df.to_parquet(cache, index=False)
        conn.close()
 
    return df

#### data cleaning
def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data:
      parse dates
      drop duplicate (code, date) rows
      drop rows with null dates
      sort by (code, date)
    returns new cleaned copy
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
 
    # drop duplicated (code, date) rows, keep first occurrence
    df_clean = df.drop_duplicates(keep="first")
    # drop rows with null dates (found 4 in the original data)
    df_clean = df_clean.drop(df_clean[df_clean["date"].isnull()].index)
    # sort again just in case
    df_clean = df_clean.sort_values(["code", "date"])
 
    return df_clean


def drop_known_bad_date(df_clean: pd.DataFrame) -> pd.DataFrame:
    '''
    drop the known bad date (2000-09-05, a parsing artifact for 000905.SH)

    seperate this out because it is a specific step for this particular data set
    '''
    df_clean = df_clean.drop(df_clean[df_clean["date"] == "2000-09-05"].index)

    return df_clean


#### validation after cleaning (coz time sequence is important here)
def check_monotonic_dates(df: pd.DataFrame) -> dict:
    """
    Diagnostic: check whether dates are strictly increasing. 
    Returns a dict of {code: problem_rows_df} for any codes
    that fail the check (empty dict if all pass).
    """
    issues = {}
    for code in df["code"].unique():
        sub = df[df["code"] == code].sort_values("date")
        if not sub["date"].is_monotonic_increasing:
            diffs = sub["date"].diff()
            issues[code] = sub[diffs <= pd.Timedelta(0)]
    return issues


#### unit adjustment (a very specific step for this particular data)

# show the dates so it can be generalised further
fix_code_dates = {
    "932000.CSI": ["2025-08-04", "2025-08-05", "2025-08-06", "2025-08-07", "2025-08-08", "2025-08-11"],
    "000985.CSI": ["2025-08-04", "2025-08-05", "2025-08-06", "2025-08-07", "2025-08-08", "2025-08-11"],
    "000300.SH": ["2025-08-04", "2025-08-05", "2025-08-06", "2025-08-07", "2025-08-08", "2025-08-11"],
}

fix_code_from_date = {
    "399303.SZ": "2025-08-04",  # multiply everything from this date onward
}
 
# fix unit in a specific range of dates
def adjust_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the known volume unit-scale fixes identified during data inspection.
    Certain codes had VOLUME under-reported by 1,000,000 on specific dates.
    """
    df = df.copy()
 
    for code, dates in fix_code_dates.items():
        mask = (df["code"] == code) & (df["date"].isin(pd.to_datetime(dates)))
        df.loc[mask, "VOLUME"] *= 1_000_000
 
    for code, from_date in fix_code_from_date.items():
        mask = (df["code"] == code) & (df["date"] >= from_date)
        df.loc[mask, "VOLUME"] *= 1_000_000
 
    return df

#### export to csv file (Qlib expected format)
def export_for_qlib(df: pd.DataFrame, output_dir: str = "cleaned_raw_data"):
    """
    Write one CSV per instrument, in the OHLCV(+factor, vwap) format
    """
    os.makedirs(output_dir, exist_ok=True)
 
    for code, group in df.groupby("code"):
        out = group[["date", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]].copy()
        out.columns = ["date", "open", "high", "low", "close", "volume"]
        out["factor"] = 1.0  # placeholder if no real adjustment factor is available
        out["vwap"] = group["AMT"] / group["VOLUME"]

        out = out.sort_values("date")
        out_path = os.path.join(output_dir, f"{code}.csv")
        out.to_csv(out_path, index=False)

