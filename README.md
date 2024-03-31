# stock-rl
Reinforcement Learning on stock data

## 股价数据
- 从 trainset.zio 解压出来
```
Unnamed: 0 - 这是索引列，用于标识每行数据。在投资环境中，它可以帮助我们跟踪每个观察结果，但是对于投资决策本身并无直接影响。
ticker - 股票的代码或符号，代表的是我们正在投资的特定公司。在投资环境中，它可以帮助我们区分不同公司的股票。
date - 观察日期，在投资环境中，它可以帮助我们跟踪股票价格随时间的变化，并据此做出投资决策。
open - 在观察日期开盘时的价格，它可以帮助我们了解市场在开盘时的情况。
high - 在观察日期内的最高价格，它可以帮助我们了解市场在该日的最高点情况。
low - 观察日期内的最低价格，它可以帮助我们了解市场在该日的最低点情况。
close - 在观察日期收盘时的价格，它是衡量当日股票表现的重要指标。
volume - 在观察日期交易的股票数量，它可以帮助我们了解市场流动性的情况。如果成交量很大，可能表示市场上对该股票的兴趣很高。
outstanding_share - 这是公司当前未卖出的股票数量，它可以帮助我们了解公司的规模。
turnover - 这是股票在观察日期内的换手率，它可以帮助我们了解市场活跃度。如果换手率很高，可能意味着市场上对该股票的交易活动很频繁。
```

## 新闻数据

已上传kaggle，有纯文本的，也有使用 `intfloat/multilingual-e5-large` 进行文档向量化后的数据。

## 策略
单一证券的价值不应超过投资组合总价值的 10%。

组合切换，在上一次的基础上切换投资组合

单日最大交易量：
这个参数可以根据你的虚拟环境的规模和场景进行设定。
现实中，交易量可以非常大，比如某些大盘股每天的交易量可能达到数百万或者数千万股。
在虚拟环境中，你可以设定一个合理的最大交易量，比如10000股或者100000股。

描述词	交易量
很多	9999 股
多	    7000 股
普通	5000 股
少	    3000 股
很少	1000 股


是否可以以high和low的均值代表当天价格：
原则上，你可以使用high和low的平均值来代表当天的价格。
这可能是一个合理的假设，因为它认为价格在最高点和最低点之间均匀地波动。
但是要注意，这只是一个简化的模型，可能不能完全代表真实的市场情况。
例如，在一个震荡剧烈的交易日，真实的平均价格可能会偏离high和low的平均值。
而且，这个假设也忽略了价格的时间序列信息，即价格如何随着时间的推移而变化。

在环境刷新时，目前的投资组合应该分配在最能赚钱的地方。
不能梭哈，每个项目的资金占比不能超过总投资的10%。（消融实验）

state:
- 当前投资组合
- 历史
- news embedding

reward:
- 环境刷新到下一天，根据 (前后两天价格差*持仓) 给予 reward
- 不用等刷新到下一天，根据投资方案和 max_trend 的符合程度给予 reward（supervised training）

action:
- 对于每个股票，决定投多少比例的总资产
- 设置变更标的所需的阈值，投资组合不要变化过于杂乱，小的变化就不要换了，减少手续费
- 环境判断当前volumn是否够