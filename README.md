# official_run
Running multiple LMs on a multi-stock (actually index) data set to predict future returns.
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
