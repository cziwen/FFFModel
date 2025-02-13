import sys
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import matplotlib.pyplot as plt
import json
import talib

def fetch_multi_feature_data_ (ticker, start_date, end_date,
                              features=["Close", "Open", "High", "Low", "Volume"]):
    """
    获取多种特征 (开高低收+成交量等) 的日级别历史数据

    :param ticker: 股票代码 (如 "AAPL")
    :param start_date: 开始日期 (如 "2022-01-01")
    :param end_date: 结束日期 (如 "2024-02-01")
    :param features: 想要获取的列名称列表
    :return: 包含所需列的 pandas.DataFrame，索引为日期
    """
    data = yf.download (ticker, start=start_date, end=end_date, interval="1d")
    # data 包含 [Open, High, Low, Close, Volume]

    # 如果某些列不存在或拼写错误，会抛出 KeyError，你可以做个简单的保护
    return data[features].values

import yfinance as yf
import pandas as pd


def fetch_multi_feature_data (
        ticker: str,
        start_date: str,
        end_date: str,
        features=["Close", "Open", "High", "Low", "Volume"]
) -> pd.DataFrame:
    """
    获取多种特征 (开高低收+成交量等) 的日级别历史数据，并增加一个表示“星期几”的新列（0=Monday, 4=Friday）。
    同时会剔除周六、周日的行（若存在）。

    :param ticker: 股票代码 (如 "AAPL")
    :param start_date: 开始日期 (如 "2022-01-01")
    :param end_date: 结束日期 (如 "2024-02-01")
    :param features: 想要获取的列名称列表（默认 [Close, Open, High, Low, Volume]）
    :return: 包含所需列的 pandas.DataFrame，索引为交易日 (datetime)，
             最后多一列 'Weekday' 表示周几 (0=Monday, 4=Friday)。
    """

    # 1) 从 yfinance 下载数据
    data = yf.download (ticker, start=start_date, end=end_date, interval="1d")

    # 2) 过滤无效特征列，避免 KeyError
    # missing_features = set (features) - set (data.columns)
    # if missing_features:
    #     print (f"Warning: 以下列在下载的数据中不存在，将被忽略: {missing_features}")
    # valid_features = [f for f in features if f in data.columns]

    # 3) 剔除周六(5) 和 周日(6)（以防某些市场数据异常）
    data = data[data.index.dayofweek < 5]

    # 4) 添加 Weekday 列（0=Monday, 4=Friday）
    data["Weekday"] = data.index.dayofweek

    # 5) 返回所需特征列 + 'Weekday'
    return data[features + ["Weekday"]]


def fetch_intraday_data (ticker, period="7d", interval="15m"):
    """
    获取指定周期和时间间隔的日内 (Intraday) 数据

    :param ticker: 股票代码 (如 "AAPL")
    :param period: 需要获取的数据范围 (例如 "1d", "5d", "7d", "30d", "60d" etc.)
    :param interval: 间隔, 可选 "1m", "2m", "5m", "15m", "30m", "1h" ...
    :return: pandas.DataFrame (包含 Open, High, Low, Close, Volume, Adj Close 等)
    """
    data = yf.download (ticker, period=period, interval=interval)
    return data


def create_sequences (data, seq_length):
    """
    创建时间序列数据集，每个样本包含 `seq_length` 天的【多特征输入】，
    目标是预测第 `seq_length` 天的【多特征】。

    参数:
    -------
    data : np.ndarray
        形状 (N, F)，N 表示时间序列长度，F 表示特征数。

    seq_length : int
        LSTM 输入序列的长度 (窗口大小)。

    返回:
    -------
    X_Tensor: torch.FloatTensor
        形状 (样本数, seq_length, F)，对应 LSTM 输入。

    y_Tensor: torch.FloatTensor
        形状 (样本数, F)，对应预测目标 (单步、多特征)。
    """
    X, y = [], []
    num_samples = len (data)

    for i in range (num_samples - seq_length):
        # 取前 seq_length 天 => shape (seq_length, F)
        X.append (data[i: i + seq_length])

        # 取第 seq_length 天（这里是单步预测，但包含全部特征）=> shape (F,)
        y.append (data[i + seq_length])

    # 转为 numpy 再转 torch
    X = np.array (X)  # shape: (样本数, seq_length, F)
    y = np.array (y)  # shape: (样本数, F)

    X_Tensor = torch.tensor (X, dtype=torch.float32)
    y_Tensor = torch.tensor (y, dtype=torch.float32)

    return X_Tensor, y_Tensor


def plot_stock_prediction (actual, predicted, ticker="Unknown"):
    """
    绘制股票预测结果，x 轴使用交易日序号 (0, 1, 2, ...) 而不是日期。

    :param actual: 实际价格 (array 或 list)
    :param predicted: 预测价格 (array 或 list)
    :param ticker: 股票代码（用于标题）
    """
    plt.figure (figsize=(10, 5))

    # 生成 x 轴交易日序号 (0, 1, 2, ...)
    x_axis = range (len (actual))

    # 绘制实际价格和预测价格
    plt.plot (x_axis, actual, label="Actual", color='blue')
    plt.plot (x_axis, predicted, label="Predicted", color='red', linestyle='dashed')

    # 设置标签
    plt.xlabel ("Trading Days")
    plt.ylabel ("Stock Price")
    plt.title (f"LSTM {ticker} Price Prediction")
    plt.legend ()

    # 显示图像
    plt.show ()


def load_config (config_path="CONFIG.json"):
    """
    从 JSON 配置文件中加载 CONFIG 字典
    :param config_path: JSON 配置文件路径
    :return: 配置字典
    """
    with open (config_path, "r", encoding="utf-8") as file:
        config = json.load (file)
    return config


def print_progress_bar (progress, content=""):
    '''
    :param progress: 百分比进度
    :param content: 需要显示的内容
    '''
    sys.stdout.write (f"\r进度[{int (progress)}/{100}] - {content}")
    sys.stdout.flush ()
