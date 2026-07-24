# official_run
Running multiple LMs on a multi-stock (actually index) data set to predict future returns.

# multi_stock_full_code:
- the outline:
    - insert data
    - first round of data cleaning
    - insert Alpha158 and Ridge training
    - abnormal predict results
    - second round of data cleaning
    - insert Alpha158
    - Ridge
    - LightGBM
    - MLP
    - LSTM
- notes:
    - the second round of data cleaning could be avoided if i had checked the mathematical summary of the numeric inputs
    - the raw data itself changed unit without notification
    - the most basic model produced the best result, perhaps this data set (9 stocks span 7 years) is still too small?
    - overfitting issue significantly improved compared to test_run project
 
# multi_stock_pipeline:
Stock Return Prediction Pipeline:

A quantitative machine learning pipeline for predicting stock returns, built during a summer internship. The pipeline covers the full workflow from raw market data to model evaluation and comparison — data ingestion, feature engineering, model training, and walk-forward validation.

- Overview:

This project explores stock return prediction using a progressively escalating set of models, evaluated under realistic, time-aware validation conditions. Rather than a single train/test split, the pipeline uses walk-forward validation across sequential time periods to better approximate how a model would perform if deployed over time.

- Pipeline Structure
    - Data Ingestion & Cleaning Raw market data is ingested and cleaned using Qlib, Microsoft's quantitative investment platform.
    - Feature Engineering Features are generated using Qlib's Alpha158 feature set, along with custom preprocessing implemented in NumPy,     including outlier-robust scaling and cross-sectional ranking of features.
    - Model Escalation Ladder Models are trained and compared in increasing order of complexity:
    - Ridge Regression: linear baseline
    - LightGBM: gradient-boosted trees
    - MLP: multi-layer perceptron
    - LSTM: sequence model for capturing temporal patterns across stocks
    - Validation Walk-forward validation is used to evaluate each model on sequential out-of-sample periods, giving a more realistic picture of performance than a single static split.
    - Model Stability Analysis (Seed sensitivity analysis to check consistency of results across random initializations)
    - Comparison of ensembling versus single-model selection
    - Hyperparameter tuning via Optuna
    - Evaluation Metrics:

Models are compared using:

MSE / MAE
Pearson correlation
Rank IC (Information Coefficient)

A zero-baseline model is included as a reference point for comparison.


Background Work:

Before building the Qlib-based pipeline, core deep learning architectures (MLP, LSTM) were implemented from scratch in NumPy, including manual backpropagation and gate-by-gate LSTM computation, as a way to build a solid understanding of the underlying mechanics before relying on higher-level frameworks.

Tech Stack:

Python (pandas, NumPy, scikit-learn)
Qlib for data handling and feature engineering
LightGBM
PyTorch
Optuna for hyperparameter tuning


Next Steps:

Portfolio-level backtest using Ridge as the stable baseline model

- Notes: This is an internship project. Data files are not included in this repository.
