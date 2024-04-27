import pandas as pd
from tqdm import tqdm
import typing as tp
import os
import pickle

def readDataset(name='trainset.txt', embed_dir=None):
    with open(name, 'r', encoding='utf-8') as f:
        fpaths = [fpath for fpath in f.read().split('\n') if fpath != '']

    # 初始化2个空的list，用于存储价格、成交量
    # Initialize lists to hold the data for each DataFrame
    if embed_dir:
        full_embeds = {}
        subfiles = os.listdir(embed_dir)
        for file in tqdm(subfiles, desc='loading embeds'):
            file_dir = os.path.join(embed_dir, file)
            with open(file_dir, 'rb') as f:
                emb_dict = pickle.load(f)
            for k, emb in emb_dict.items():
                simple_k = k[:10]
                if full_embeds.get(simple_k, None) is not None:
                    continue  # 只需要当天的第一个新闻
                else:
                    full_embeds[simple_k] = emb  # 1d array, float32

    prices_data, volumes_data, embeds_data = [], [], []
    for fpath in tqdm(fpaths):  # 循环处理文件路径列表
        # 读取CSV文件到DataFrame，并设置日期为索引
        df = pd.read_csv(fpath, index_col='date', parse_dates=['date'])
        df = df[df.index > '2016-01-01']

        # 提取股票代码作为列名
        ticker = fpath.split('\\')[-1].split('.')[0]
        
        # 将价格、趋势和成交量添加到相应的DataFrame中
        # Append the series with the ticker as the name to the lists
        prices_data.append(df['price'].rename(ticker))
        volumes_data.append(df['volume'].rename(ticker))
        if embed_dir:
            for index, _ in df.iterrows():
                date_index = str(index)[:10]
                daily_embed = full_embeds.get(date_index, np.random.randn(1024))
                embeds_data.append(daily_embed)

    # 将所有DataFrame的索引统一，确保日期对齐，缺失的数据用前一个值填充
    # Concatenate all data into their respective DataFrames
    prices_df = pd.concat(prices_data, axis=1)
    volumes_df = pd.concat(volumes_data, axis=1)

    # Reindex and fill missing dates
    all_dates = pd.date_range(start=prices_df.index.min(), end=prices_df.index.max(), freq='D')
    prices_df = prices_df.reindex(all_dates).ffill()
    volumes_df = volumes_df.reindex(all_dates).ffill()

    # Fill any remaining missing values with 0
    prices_df.fillna(0, inplace=True)
    volumes_df.fillna(0, inplace=True)

    if embed_dir:
        return prices_df, volumes_df, embeds_data
    else:
        return prices_df, volumes_df


import numpy as np
from game_env import readDataset
 

class EmulatorEnv:
    def __init__(self, initial_money=500000, start_day=10, end_day=-1, embed_dir=None):
        if embed_dir:
            self.prices_df, self.volumes_df, self.embeds_data = readDataset(embed_dir=embed_dir)
            self.use_embed = True
        else:
            self.prices_df, self.volumes_df = readDataset(embed_dir=embed_dir)
            self.use_embed = False

        self.bunch_size = 1000  # 暂时用不到

        self.initial_money = initial_money 
        self.start_day = start_day

        self.current_day = start_day            # 日期计数
        max_day = len(self.prices_df)
        self.end_day = max_day - end_day if end_day < 0 else end_day
        self.end_day = min(max_day, self.end_day)

        # self.market_value = 0         # 账户市值
        self.volumes = np.zeros(512)    # 记录持仓量，方便跟踪总市值
        self.available_cash = self.initial_money  # 账户资金
        self.asset = self.available_cash  # 账户资产 = 账户市值 + 账户资金

    def reset(self):
        # Reset the environment to the initial state
        self.current_day = self.start_day
        # self.market_value = 0
        self.volumes = np.zeros(512)
        self.available_cash = self.initial_money
        self.asset = self.available_cash
        if self.use_embed:
            return self._get_state(), self._get_embed(), 0, False
        else:
            return self._get_state(), 0, False  # reward=0, done=Falsee

    def tomorrow_prices(self):
        tomorrow = self.current_day + 1 if self.current_day < self.end_day - 1 else self.current_day
        return self.prices_df.iloc[tomorrow].values

    def step(self, qvalues: np.ndarray):
        # action is a vector of Q Values
        
        stock_prices = self.prices_df.iloc[self.current_day].values  # 由于使用均价，可以忽略交易价格滑点
        zero_price_mask = stock_prices==0  # 检查 stock_prices 中是否存在0
        stock_prices[zero_price_mask] = 1e-10
        
        # 暂时设置交易量的最小单位为1，现实环境可能不行

        # 先卖后买。 qvalue<=0 的股票必须全部卖掉
        sell_volumes = np.where(qvalues<=0, 1, 0) * self.volumes  
        self.volumes -= sell_volumes  # 减去持仓量
        self.available_cash += stock_prices.dot(sell_volumes)  # 卖掉了垃圾股票，拿钱

        # 将资产按比例分配到 qvalue>0 的股票上 or 只分配到前10上面
        sorted_qvals = sorted(qvalues, reverse=True)
        for i in range(0, 21):
            if sorted_qvals[i] < 0: break
        buy_threshold = sorted_qvals[i]  # 集合至少要有10只股票，否则要报错了

        buy_strategy = np.where(qvalues>=buy_threshold, qvalues, 0)
        buy_strategy[zero_price_mask] = 0
        if np.sum(buy_strategy) > 0:  # 总和大于零，这一天的买入才有意义
            buy_strategy /= np.sum(buy_strategy)

            buy_volumes = np.floor((self.available_cash*0.1 * buy_strategy) / stock_prices)  # 有的 stock_prices为 0，如何防止除以这些数值？
            buy_volumes = np.minimum(buy_volumes, self.volumes_df.iloc[self.current_day].values)
            self.available_cash -= buy_volumes.dot(stock_prices)  # 减去买股票花掉的钱，可能剩下些零钱
            self.volumes += buy_volumes

            current_market_value = stock_prices.dot(self.volumes)
            self.asset = current_market_value + self.available_cash
            reward = np.sum(self.tomorrow_prices() * self.volumes) - current_market_value
        else:
            reward = 0

        # 应用对称对数变换，防止训练中出现超大值，破坏稳定
        reward = np.sign(reward) * np.log1p(np.abs(reward))

        # Check if the end of the dataset is reached
        self.current_day += 1
        done = self.current_day > self.end_day - 2# or self.asset < 10
        if self.asset < -self.initial_money:  # 允许负债，但数量上不能超过原始资金
            done = True
            print('wasted')
        # Get the next state
        next_state = self._get_state()
        if self.use_embed:
            return next_state, self._get_embed(), reward, done
        else:
            return next_state, reward, done  # TODO: 另外的done: 负债超过50w

    def _calc_investment_ratio(self):
        current_stock_values = self.prices_df.iloc[self.current_day].values * self.volumes
        return current_stock_values / (self.available_cash + np.sum(current_stock_values))
    
    def _get_state(self):
        # Retrieve the state, which is the past 3-day prices of all 512 stocks
        # Concatenate investment over total investment ratio to the state
        state_prices = self.prices_df.iloc[self.current_day-3:self.current_day].values / 100 # 降低数值大小

        state_prices = state_prices.flatten()
        assert state_prices.shape == (512*3,), f"Expected shape (512*3), got {state_prices.shape}"
        # Calculate the investment to total investment ratio
        investment_ratio = self._calc_investment_ratio()  # 今天决策后的资金分配情况
        # Flatten prices and concatenate investment ratio
        # print(state_prices.shape)
        state = np.concatenate((state_prices, investment_ratio))
        return state

    def _get_embed(self):
        return self.embeds_data[self.current_day]

# Initialize the environment
if __name__ == '__main__':
    env = EmulatorEnv()
    state = env.reset()

    # Emulate a random action
    random_actions = np.random.randint(-1, 1, size=(512,))  # Example random actions, should be adjusted based on actual needs
    next_state, reward, done = env.step(random_actions)

    import matplotlib.pyplot as plt
    plt.bar(range(512), next_state[-512:])
    plt.show()

