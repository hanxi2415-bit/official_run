# official_run

这是一个使用 9 只指数数据预测未来收益率的机器学习实验项目。

## 内容

- `multi_stock_ml.ipynb`：数据读取、两轮数据清洗、Alpha158 特征生成、Ridge、LightGBM、MLP 调参、LSTM 以及最终比较。
- `*.csv`：用于 Qlib 的逐指数清洗后 OHLCV 输入数据。

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

重新运行 Notebook 前，需要在 `.env` 中填写：

```text
MYSQL_HOST=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=intern
QLIB_PROVIDER_URI_MULTI=
QLIB_PROVIDER_URI_CLEANED=
```

## 实验流程

- 插入数据。
- 第一轮数据清洗。
- 生成 Alpha158 特征并训练 Ridge。
- 排查异常预测结果。
- 针对成交量单位变化进行第二轮数据清洗。
- 重新生成 Alpha158 特征。
- 训练 Ridge、LightGBM、MLP 和 LSTM。
- 与“预测收益率恒为 0”的简单基线比较。

## 可复现性说明

- 数据库账号、密码和本地 token 应放在 `.env` 中，不要提交到 git。
- 生成的 Qlib 二进制数据建议放在 git 之外，或放到已忽略的 `qlib_data/` 目录。
- 验证集应只用于模型选择和超参数选择；测试集应保留到最终报告时一次性使用。
- 模型训练前应先完成数据质量检查：日期解析、重复 `(code, date)`、缺失日期、数值统计摘要、非正价格/成交量，以及突发单位变化。

## 当前判断

- 第二轮清洗很可能是原始成交量字段发生单位变化导致的。
- 当前结果看起来 Ridge 最稳，但这个结论还需要一次干净复跑和统一结果表支撑。
- 在该数据规模下，更复杂的模型不一定更好；比起单次测试分数，更应关注不同 seed 和滚动窗口下的稳定性。
