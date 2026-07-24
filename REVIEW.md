# 审查建议

这个仓库的探索过程是有价值的：异常预测结果最终被追溯到输入数据质量问题，这个排查方向是正确的。下一步重点应放在实验可复现，以及把验证集和测试集职责分清楚。

## 高优先级

- 将数据库连接信息移出 `multi_stock_ml.ipynb`。当前 Notebook 的连接单元里直接写了 host、user、password 和 database，建议改为从 `.env` 环境变量读取。
- 移除 `/Users/hanxi/...` 这类本机绝对路径。Qlib 导出路径建议使用仓库相对路径（如 `raw_data/`、`qlib_data/`）或环境变量。
- 在信任 MLP 调参结果前，需要先修正 Optuna objective 单元：
  - 代码预测的是 `X_valid`，但计算 `mse_mlp` 时使用了旧的 `pred_series_mlp` 和 `y_test`。
  - study 使用了 `direction="maximize"`，但返回的是 MSE。若目标是 MSE，应改为最小化；若目标是最大化，则 objective 应返回 Rank IC。
- 不要用测试集参与模型选择。当前若干 MLP 稳定性检查直接评估 `X_test`，建议改为先在验证集上选模型和超参数，最后只用测试集做最终报告。

## 数据质量建议

- 在特征生成前增加数据校验段：缺失日期、重复 `(code, date)`、非正 OHLCV、按 code 的数值统计摘要、成交量单位跳变、异常 AMT/VOLUME 比值。
- 用一张表记录成交量单位修正规则：code、日期范围、乘数、判断依据或外部验证来源。
- 如果原始数据源可能静默改变单位，建议在流程中保留断言或异常检测，让这类问题尽早失败，而不是等到模型结果异常后才发现。

## 方法论建议

- 把零收益基线和 Ridge、LightGBM、MLP、LSTM 放在同一张结果表中比较。
- 可以同时报告 pooled Pearson 和 daily Rank IC，但要明确哪个指标用于模型选择。
- 在得出“Ridge 最好”的结论前，建议增加 rolling 或 walk-forward 验证。
- seed 稳定性应先报告验证集上的均值和标准差，再用选定配置跑测试集。

## 建议下一步

抽出一个小型 pipeline 脚本或工具模块，包含：读取数据、校验数据、应用单位修正、构建 Alpha158 数据、按时间划分、统一评估预测、输出比较表。Notebook 主要保留解释和图表即可。
