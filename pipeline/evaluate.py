"""
prediction evaluation and confusion matrix 
"""


import numpy as np
import pandas as pd
 
from pipeline.partition import get_fold_data


#### single prediction evaluation
def evaluate_predictions(
        preds, 
        y_true,
        dates_test = None, # for LSTM
        codes_test = None, # for LSTM
        ) -> dict:
    """
    Compute the standard metric set used across all models in this project:
      - MSE, MAE
      - Pearson correlation (pred vs actual)
      - Rank IC: mean of the per-day Spearman correlation between pred and
        actual, grouped by the 'datetime' index level.
 
    y_true must be a pd.Series with a MultiIndex containing a 'datetime' level 
    (as produced by Alpha158 and the features functions)
    """
    if type(y_true) == np.array:
        test_index = pd.MultiIndex.from_arrays([dates_test, codes_test], names=["datetime", "instrument"])
        results = pd.DataFrame({"pred": np.asarray(preds), "actual": y_true}, index=test_index)
    else:
        results = pd.DataFrame({"pred": np.asarray(preds), "actual": y_true.values}, index=y_true.index)

    mse = ((results["actual"] - results["pred"]) ** 2).mean()
    mae = (results["actual"] - results["pred"]).abs().mean()
    pearson = results["pred"].corr(results["actual"])
 
    daily_ic = results.groupby(level="datetime").apply(
        lambda g: g["pred"].corr(g["actual"], method="spearman")
    )
    rank_ic = daily_ic.mean()
 
    return {"mse": mse, "mae": mae, "pearson corr": pearson, "rank_ic": rank_ic}

# zero-baseline comparison
def zero_baseline_metrics(y_true: pd.Series) -> dict:
    """
    Metrics for the trivial 'always predict 0' baseline, for sanity-checking
    that a model is actually better than doing nothing.
    """
    preds = np.zeros(len(y_true))
    return evaluate_predictions(preds, y_true)


#### seed stability check
"""
it is rather unnecessary to extract a new function for random seeds
because every model set seed differnetly
"""

#### walk-forward evaluation
def walk_forward_evaluate(
    implement_f,
    # a function that takes in (X_fit, y_fit, X_test) and returns pred
    # in short: a model and its fit
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple],
    best_params: dict,
    combine_train_valid: bool = True, 
    # MLP can be trained on both train and valid sets as the model training itself does not need valid
) -> pd.DataFrame:
    """
    Run walk-forward evaluation across all folds for one model.
 
    fit_predict_fn(X_fit, y_fit, X_test) -> output: preds
 
    Returns one row per fold: fold number, test_period_until, and the
    standard metric set.
    """
    fold_results = []
    i = 0

    for split in splits:
        X_train_r, y_train_r, X_valid_r, y_valid_r, X_test_r, y_test_r = get_fold_data(X, y, split)
 
        if combine_train_valid: # Ridge and MLP
            X_fit = pd.concat([X_train_r, X_valid_r], axis=0)
            y_fit = pd.concat([y_train_r, y_valid_r], axis=0)
            preds, matrix = implement_f(X_fit, y_fit, X_test_r, y_test_r, best_params,)

        else: # LightGBM and LSTM
            preds, matrix = implement_f(X_train_r, y_train_r, X_valid_r, y_valid_r, X_test_r, y_test_r, best_params, )

        i += 1
        matrix["fold"] = i
        matrix["test_period_until"] = split[2]
        fold_results.append(matrix)
 
        print(f"Fold {i}: test={split[2]}, rank_ic={matrix['rank_ic']:.5f}")
    
    print(f"\nMean rank_ic across folds: {fold_results['rank_ic'].mean():.5f}")
    print(f"Std rank_ic across folds: {fold_results['rank_ic'].std():.5f}")
 
    return pd.DataFrame(fold_results)[["fold", "test_period_until", "mse", "mae", "pearson", "rank_ic"]]


#### cross model comnparison table 
def compare_models(combined_fold_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Given {"Ridge": walk_forward_df, "LightGBM": ..., "MLP": ..., "LSTM": ...},
    return one row per model with mean/std for each metric across folds.
    """
    rows = []
    for name, df in combined_fold_results.items():
        row = {"model": name}
        for col in ["mse", "mae", "pearson", "rank_ic"]:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()
        rows.append(row)
 
    return pd.DataFrame(rows).set_index("model")


