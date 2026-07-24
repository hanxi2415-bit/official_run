
"""
Model definitions and parameter tuning
getting the final prediction
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

import lightgbm as lgb

import torch
import torch.nn as nn
 
from pipeline.evaluate import evaluate_predictions
from pipeline.partition import build_sequences


#### Ridge

def tune_ridge_alpha(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    alphas=None,
) -> float:
    """
    Grid search over alpha values for Ridge regression, selecting the one
    with lowest validation MSE. Returns the best alpha.

    Ridge has only one parameter so there's no need to implement optuna
    """
    if alphas is None:
        alphas = np.logspace(-1, 3, 20)
 
    best_alpha, best_score = None, np.inf
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="cholesky")
        model.fit(X_train, y_train.values.ravel())
        preds = model.predict(X_valid)
        score = ((y_valid.values.ravel() - preds) ** 2).mean()
        if score < best_score:
            best_alpha, best_score = alpha, score
 
    return best_alpha

def implement_ridge(
        X_fit: pd.DataFrame, 
        y_fit: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        best_alpha: float,
        ):
    final_model = Ridge(alpha=best_alpha, solver="cholesky")
    final_model.fit(X_fit, y_fit.values.ravel())

    preds = final_model.predict(X_test)
    matrix = evaluate_predictions(preds, y_test)

    return preds, matrix



#### optuna tuning
def tune_with_optuna(objective, n_trials: int = 50, direction: str = "maximize") -> dict:
    """
    Thin wrapper around an Optuna study: you supply objective(trial) 
    this handles study creation/optimization and returns study.best_params
    """
    import optuna
 
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params



#### LightGBM

# LightGBM tuning
def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int = 50,
) -> dict:
    """
    Optuna search over LightGBM's num_leaves, learning_rate, max_depth,
    min_child_samples -- optimizing validation rank IC.
    """
    train_set = lgb.Dataset(X_train, label=y_train, params={"feature_pre_filter": False})
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
 
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mse",
            "verbose": -1,
            "seed": 42,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        model = lgb.train(
            params, train_set, num_boost_round=500,
            valid_sets=[valid_set], ##### here's a question id like to mark down
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        preds = model.predict(X_valid) #### because we are using it again
        return evaluate_predictions(preds, y_valid)["rank_ic"]
 
    return tune_with_optuna(objective, n_trials=n_trials)


# getting pred
def implement_lgb(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        best_params: dict,
        ):

    # merge tuned params with the fixed base config
    params = {
        "objective": "regression",
        "metric": "mse",
        "verbose": -1,
        "seed": 42,
        **best_params,   # unpacks best_params' keys into this dict
    }

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    lgb_model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
    )

    preds = lgb_model.predict(X_test, num_iteration = lgb_model.best_iteration)
    matrix = evaluate_predictions(preds, y_test) 
    return preds, matrix


#### MLP

# mlp tuning on optuna
def tune_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int = 50,
    seed: int = 42,
) -> dict:
    """
    Optuna search over MLPRegressor architecture (1-3 layers, 16-128 units
    each), alpha, and learning_rate_init -- optimizing validation rank IC.
    """
 
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
 
        model = MLPRegressor(
            hidden_layer_sizes = tuple(trial.suggest_int(f"units_l{i}", 16, 128, step=16) for i in range(n_layers)),
            alpha = trial.suggest_float("alpha", 1e-4, 0.5, log=True),
            learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            activation="relu",
            solver="adam",
            max_iter=1000,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return evaluate_predictions(preds, y_valid)["rank_ic"]
 
    return tune_with_optuna(objective, n_trials=n_trials)


# getting pred for MLP
def implement_mlp(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    best_params: dict,
    seed: int = 42,
):
    hidden_layer_sizes = tuple(
        best_params[f"units_l{i}"] for i in range(best_params["n_layers"])
    )
    for seed in seeds:
        mlp_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=best_params["alpha"],
            learning_rate_init=best_params["learning_rate_init"],
            activation="relu", solver="adam", max_iter=1000,
            early_stopping=False, random_state=seed,
        )  
    mlp_model.fit(X_fit, y_fit)

    preds = mlp_model.predict(X_test) 
    matrix = evaluate_predictions(preds, y_test) 
    return preds, matrix


# random seed stability test MLP
seeds = [0, 1, 41, 123, 7]
rank_ics = []

def seed_stability_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    best_params: dict,
    seeds: list,
):
    hidden_layer_sizes = tuple(
        best_params[f"units_l{i}"] for i in range(best_params["n_layers"])
    )
    for seed in seeds:
        mlp_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=best_params["alpha"],
            learning_rate_init=best_params["learning_rate_init"],
            activation="relu", solver="adam", max_iter=1000,
            early_stopping=False, random_state=seed,
        )     
        mlp_model.fit(X_train, y_train)
        preds = mlp_model.predict(X_valid)
        matrix = evaluate_predictions(preds, y_valid)
        rank_ics.append(matrix["rank_ic"])
        print(matrix)
    print(f"\nMean rank_ic across folds: {np.mean(rank_ics):.5f}")
    print(f"Std rank_ic across folds: {np.mean(rank_ics):.5f}")
    return rank_ics 



#### LSTM

class LSTMRegressor(nn.Module):
    """Simple single-output LSTM regressor over a sequence of features."""
 
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
 
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # final layer's hidden state
        return self.fc(last_hidden).squeeze(-1)
    

feature_cols = ["log_ret", "vol_z", "hl_range", "STD20", "RESI30", "IMIN20"]

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    input_size: int = len(feature_cols),
    hidden_size: int = 32,
    num_layers: int = 1,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int = 20,
    seed: int | None = None,
) -> LSTMRegressor:
    """
    Train an LSTMRegressor with manual early stopping on validation loss.
    Returns the model loaded with its best (lowest valid-loss) weights.
    """
    if seed is not None:
        torch.manual_seed(seed)
 
    model = LSTMRegressor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
 
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_t = torch.tensor(y_valid, dtype=torch.float32)
 
    best_valid_loss = float("inf")
    patience_counter = 0
    best_state = None
 
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
 
        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(X_valid_t), y_valid_t).item()
 
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
 
    model.load_state_dict(best_state)
    return model
 
 
 # get pred
def predict_lstm(model: LSTMRegressor, X_test: np.ndarray) -> np.ndarray:
    """Run inference with a trained LSTMRegressor."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        pred = model(X_t).numpy()
    return pred


# tuning LSTM with optuna
def tune_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    dates_valid,
    codes_valid,
    n_trials: int = 30,
) -> dict:
    """
    Optuna search over LSTM hidden_size, num_layers, learning rate, and
    weight_decay -- optimizing validation rank IC. Fewer default trials
    than tune_mlp/tune_lightgbm since each trial trains a full network.
    """
 
    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 16, 128, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 2)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
 
        model = train_lstm(
            X_train, y_train, X_valid, y_valid,
            hidden_size=hidden_size, num_layers=num_layers,
            lr=lr, weight_decay=weight_decay, seed=42,
        )
        preds = predict_lstm(model, X_valid)
        return evaluate_predictions(preds, y_valid, dates_valid, codes_valid)["rank_ic"]
 
    return tune_with_optuna(objective, n_trials=n_trials)


# random seed stability test LSTM
seeds = [0, 1, 41, 123, 7]
rank_ics = []

def seed_stability_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    dates_valid,
    codes_valid,
    seeds: list, 
    best_params: dict):
    for seed in seeds:
        model = train_lstm(
                X_train, y_train, X_valid, y_valid,
                **best_params, seed=seed,
            )
        preds = predict_lstm(model, X_valid)
        matrix = evaluate_predictions(preds, y_valid, dates_valid, codes_valid)
        rank_ics.append(matrix["rank_ic"])
        print(matrix)
    print(f"\nMean rank_ic across folds: {np.mean(rank_ics):.5f}")
    print(f"Std rank_ic across folds: {np.mean(rank_ics):.5f}")
    return rank_ics


# implement LSTM on best params
def implement_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: dict,
    ):

    X_train_seq, y_train_seq, dates_train_seq, codes_train_seq = build_sequences(X_train, y_train, seq_len = 20)
    X_valid_seq, y_valid_seq, dates_valid_seq, codes_valid_seq = build_sequences(X_valid, y_valid, seq_len = 20)
    X_test_seq, y_test_seq, dates_test_seq, codes_test_seq = build_sequences(X_test, y_test, seq_len = 20)

    model = train_lstm(
                X_train_seq, y_train_seq, X_valid_seq, y_valid_seq,
                **best_params, seed = 42)
    preds = predict_lstm(model, X_test_seq)
    matrix = evaluate_predictions(preds, y_test, dates_test_seq, codes_test_seq) 
    return preds, matrix



