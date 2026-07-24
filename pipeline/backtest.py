
import pandas as pd

def get_return_top_stocks(k: int, preds, y_actual):
    pred_return = []
    weight_norm = sum(range(1, k + 1))  # normalizer for rank weights: k+(k-1)+...+1

    preds = pd.DataFrame({
    "datetime": y_actual.index.get_level_values('datetime'),
    "instrument": y_actual.index.get_level_values('instrument'),
    "preds": preds
    })

    for date, pred in preds.groupby('datetime'):
        top_stock = pred.sort_values('preds', ascending=False).head(k)

        pred_return_tdy = 0
        for i, instrument in enumerate(top_stock['instrument']):
            weight = (k - i) / weight_norm
            pred_return_tdy += weight * y_actual.loc[date, instrument]

        pred_return.append({"date": date, "portfolio_return": pred_return_tdy})

    return pd.DataFrame(pred_return)



def get_benchmark(y_actual):
    benchmark = []

    for date, act_return in y_actual.groupby('datetime'):
        benchmark_tdy = act_return.mean()  # equal-weight average = mean
        benchmark.append({"date": date, "benchmark_return": benchmark_tdy})

    return pd.DataFrame(benchmark)


def compare_res(pred_return, benchmark, init_val: int = 1):

    pred_return = pred_return.sort_values('date')
    benchmark = benchmark.sort_values('date')

    est_gain = init_val
    est_gain_series = []
    for return_tdy in pred_return['portfolio_return']:
        est_gain = est_gain * (1 + return_tdy)
        est_gain_series.append(est_gain)

    bench_gain = init_val
    bench_gain_series = []
    for bench_tdy in benchmark['benchmark_return']:
        bench_gain = bench_gain * (1 + bench_tdy)
        bench_gain_series.append(bench_gain)

    print('estimated gain:', est_gain, 'benchmark gain:', bench_gain, 'portfolio gain:', est_gain - bench_gain)

    return est_gain_series, bench_gain_series

