"""
Reusable charts for the pipeline.
 
Charts specific to one-off data investigation (e.g. the volume-fix diagnostic plots)
are intentionally NOT here

the notebook should retain its explanatory charts.
"""
 
import matplotlib.pyplot as plt
import pandas as pd

def plot_actual_vs_predicted(y_true: pd.Series, preds, title: str = "Actual vs Predicted"):
    """
    Scatter plot of actual vs predicted values over time.
    y_true must have a MultiIndex with a 'datetime' level.
    """
    dates = y_true.index.get_level_values("datetime")
 
    plt.figure(figsize=(12, 5))
    plt.scatter(dates, y_true.values, s=3, label="Actual", alpha=0.7, linewidth=0.8)
    plt.scatter(dates, preds, s=3, label="Predicted", alpha=0.7, linewidth=0.8)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()