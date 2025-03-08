import sys
import json
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests


def fetch_multi_feature_data (ticker, interval, start_date, end_date,
                              features=["Close", "Open", "High", "Low", "Volume"]):
    """
    下载多特征历史数据，并添加 'Weekday' 列表示星期几。

    :param ticker: 股票代码 (如 "AAPL")
    :param interval: 数据间隔 (如 "15m")
    :param start_date: 开始日期 (如 "2022-01-01")
    :param end_date: 结束日期 (如 "2024-02-01")
    :param features: 所需特征列表
    :return: 包含特征和 'Weekday' 列的 DataFrame
    """
    data = yf.download (ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)
    if data.empty:
        raise ValueError (f"未能获取 {ticker} 从 {start_date} 到 {end_date} 的数据。")
    # data["Weekday"] = data.index.dayofweek
    # return data[features + ["Weekday"]]
    return data[features]


def fetch_intraday_data (ticker, period="7d", interval="15m"):
    """
    下载日内数据。

    :param ticker: 股票代码
    :param period: 数据周期 (如 "7d")
    :param interval: 数据间隔 (如 "15m")
    :return: 下载的 DataFrame
    """
    return yf.download (ticker, period=period, interval=interval)


def create_sequences (data, seq_length):
    """
    将数组数据转换为序列样本，用于 LSTM 输入和目标。
    整体行数会 减少 seq_length

    :param data: np.ndarray, 形状 (N, F)，N 是时间步数，F 是特征
    :param seq_length: 序列长度
    :return: X_Tensor (shape: (samples, seq_length, F)), y_Tensor (shape: (samples, F))
    """
    X, y = [], []
    for i in range (len (data) - seq_length):
        X.append (data[i: i + seq_length])
        y.append (data[i + seq_length])
    return torch.tensor (np.array (X), dtype=torch.float32), torch.tensor (np.array (y), dtype=torch.float32)


def plot_stock_prediction (actual, predicted, config, ma_window=10):
    """
    绘制股票预测结果，x 轴使用交易日序号，并额外绘制移动平均线。

    :param actual: 实际价格列表
    :param predicted: 预测价格列表
    :param config: 配置字典，包含 'interval' 和 'ticker'
    :param ma_window: 移动平均的窗口大小，默认为 10
    """
    plt.figure (figsize=(10, 5))
    x_axis = range (len (actual))
    plt.plot (x_axis, actual, label="Actual", color='blue')
    plt.plot (x_axis, predicted, label="Predicted", color='red', linestyle='dashed')

    # 计算移动平均线
    actual_series = pd.Series (actual)
    moving_avg = actual_series.rolling (window=ma_window).mean ()
    plt.plot (x_axis, moving_avg, label=f"MA({ma_window})", color='green', linestyle=':')

    plt.xlabel (f"Trading Interval {config['interval']}")
    plt.ylabel ("Stock Price")
    plt.title (f"LSTM {config['ticker']} Price Prediction")
    plt.legend ()
    plt.show ()


def plot_loss_curve (train_losses, val_losses=None, title="Loss Curve"):
    """
    绘制训练和验证损失曲线。

    :param train_losses: 训练损失列表
    :param val_losses: 验证损失列表（可选）
    :param title: 图表标题
    """
    plt.figure (figsize=(8, 5))
    plt.plot (train_losses, label="Train Loss", color="blue")
    if val_losses is not None:
        plt.plot (val_losses, label="Validation Loss", color="red")
    plt.title (title)
    plt.xlabel ("Epoch")
    plt.ylabel ("Loss")
    plt.legend ()
    plt.show ()


def load_config (config_path="CONFIG.json"):
    """
    加载 JSON 配置文件。

    :param config_path: 配置文件路径
    :return: 配置字典
    """
    with open (config_path, "r", encoding="utf-8") as file:
        return json.load (file)


def print_progress_bar (progress, content=""):
    """
    打印进度条。

    :param progress: 进度百分比
    :param content: 显示内容
    """
    sys.stdout.write (f"\r进度[{int (progress)}/{100}] - {content}")
    sys.stdout.flush ()


def fetch_polygon_data (ticker, interval, start_date, end_date,
                        features=["c", "o", "h", "l", "v", "n"],
                        api_key=None, multiplier=None, timespan=None):
    """
    通过 polygon.io API 下载历史数据，并返回包含指定特征的 DataFrame。

    :param ticker: 股票代码 (如 "AAPL")
    :param interval: 数据间隔 (例如 "1d", "15m", "1h")，用于解析 multiplier 与 timespan
    :param start_date: 开始日期 (格式 "YYYY-MM-DD")
    :param end_date: 结束日期 (格式 "YYYY-MM-DD")
    :param features: 希望返回的字段列表，默认为 ["c", "o", "h", "l", "v"]
                     分别代表 close、open、high、low 和 volume
    :param api_key: polygon.io 的 API key（必填）
    :param multiplier: (可选) 聚合倍数；若未提供，则根据 interval 自动解析
    :param timespan: (可选) 时间单位，如 "day", "minute", "hour"；若未提供，则根据 interval 自动解析
    :return: 包含指定特征的 pandas DataFrame
    """
    if api_key is None:
        raise ValueError ("必须提供 polygon.io 的 API key。")

    # 自动解析 multiplier 与 timespan，如果未指定
    if multiplier is None or timespan is None:
        if interval.endswith ("d"):
            timespan = "day"
            multiplier = int (interval[:-1])
        elif interval.endswith ("m"):
            timespan = "minute"
            multiplier = int (interval[:-1])
        elif interval.endswith ("h"):
            timespan = "hour"
            multiplier = int (interval[:-1])
        else:
            raise ValueError ("无法解析 interval，请明确提供 multiplier 和 timespan 参数。")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
        "session": "regular"  # 只返回常规交易时段数据
    }
    response = requests.get (url, params=params)
    if response.status_code != 200:
        raise Exception (f"获取数据失败: {response.text}")
    data = response.json ()
    if "results" not in data:
        raise Exception (f"返回结果异常: {data}")

    df = pd.DataFrame (data["results"])
    # 将时间戳转换为 datetime（字段 "t" 为毫秒时间戳）
    df["t"] = pd.to_datetime (df["t"], unit="ms")
    df.set_index ("t", inplace=True)
    # 重命名常用字段，方便与 yfinance 数据对应
    rename_map = {"c": "Close", "o": "Open", "h": "High", "l": "Low", "v": "Volume", "n": "Num_transactions"}
    df.rename (columns=rename_map, inplace=True)

    # 保留用户指定的字段（若 features 中给的是原始名称，则转换为重命名后的名称）
    cols_to_keep = []
    for feat in features:
        if feat in rename_map:
            cols_to_keep.append (rename_map[feat])
        elif feat in df.columns:
            cols_to_keep.append (feat)
    df = df[cols_to_keep]
    # df["Weekday"] = df.index.dayofweek
    return df


def save_dataframe (df, file_path="raw_data.csv"):
    """
    保存 DataFrame 至指定路径的 CSV 文件。

    :param df: pandas DataFrame
    :param file_path: 保存文件路径
    """
    df.to_csv (file_path, index=True)
    print (f"DataFrame 已保存至 {file_path}")