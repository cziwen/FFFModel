import pandas as pd
import numpy as np


def add_column (csv_file, column_name, default_value=None, output_file=None):
    df = pd.read_csv (csv_file)
    df[column_name] = default_value  # 添加新列，默认值可以为空或自定义
    output_file = output_file or csv_file  # 直接覆盖或另存
    df.to_csv (output_file, index=False)
    print (f"Column '{column_name}' added to {output_file}.")


def remove_column (csv_file, column_name, output_file=None):
    df = pd.read_csv (csv_file)
    if column_name in df.columns:
        df.drop (columns=[column_name], inplace=True)
        output_file = output_file or csv_file
        df.to_csv (output_file, index=False)
        print (f"Column '{column_name}' removed from {output_file}.")
    else:
        print (f"Column '{column_name}' not found in {csv_file}.")

def concatenate_csv(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df_concat = pd.concat([df1, df2], ignore_index=True)  # 纵向拼接
    df_concat.to_csv(output_file, index=False)
    print(f"Files '{file1}' and '{file2}' concatenated into {output_file}.")

def calculate_rsi(csv_file, column_name="Close", period=14):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 检查指定列是否存在
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in {csv_file}")

    # 获取收盘价数据（numpy 数组）
    close_prices = df[column_name].to_numpy()

    # 计算价格变化
    delta = np.diff(close_prices)

    # 计算涨跌额和跌幅额
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)

    # 初始化平均涨跌幅
    avg_gain = np.zeros_like(close_prices)
    avg_loss = np.zeros_like(close_prices)

    # 计算初始均值（第一窗口内的平均值）
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    # 使用加权移动平均计算 RSI
    for i in range(period + 1, len(close_prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    # 避免除零错误
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # 处理前期无法计算的 RSI 值
    rsi[:period] = np.nan

    return rsi.tolist()


def calculate_ema(prices, period):
    """ 计算指数移动平均 (EMA) """
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices, dtype=np.float64)
    ema[0] = prices[0]  # 初始化第一期的 EMA
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_macd(csv_file, column_name="Close", short_period=7, long_period=14, signal_period=5):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 检查列是否存在
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in {csv_file}")

    # 获取收盘价数据
    close_prices = df[column_name].to_numpy()

    # 计算短期 EMA 和长期 EMA
    short_ema = calculate_ema(close_prices, short_period)
    long_ema = calculate_ema(close_prices, long_period)

    # 计算 MACD 线
    macd_line = short_ema - long_ema

    # 计算信号线（MACD 线的 9 周期 EMA）
    signal_line = calculate_ema(macd_line, signal_period)

    # 计算 MACD 柱状图
    macd_histogram = macd_line - signal_line

    return macd_line.tolist(), signal_line.tolist(), macd_histogram.tolist()


def calculate_stochastic_oscillator (csv_file, close_column="Close", high_column="High", low_column="Low", period=14,
                                     signal_period=5):
    # 读取 CSV 文件
    df = pd.read_csv (csv_file)

    # 检查是否包含所需的列
    for col in [close_column, high_column, low_column]:
        if col not in df.columns:
            raise ValueError (f"Column '{col}' not found in {csv_file}")

    # 提取收盘价、最高价、最低价
    close_prices = df[close_column].to_numpy ()
    high_prices = df[high_column].to_numpy ()
    low_prices = df[low_column].to_numpy ()

    # 计算 %K 线
    k_values = np.zeros_like (close_prices, dtype=np.float64)

    for i in range (period - 1, len (close_prices)):
        highest_high = np.max (high_prices[i - period + 1:i + 1])  # N周期最高价
        lowest_low = np.min (low_prices[i - period + 1:i + 1])  # N周期最低价
        k_values[i] = (close_prices[i] - lowest_low) / (
                    highest_high - lowest_low) * 100 if highest_high != lowest_low else 0

    # 计算 %D 线（%K 线的 SMA）
    d_values = np.zeros_like (k_values, dtype=np.float64)

    for i in range (signal_period - 1, len (k_values)):
        d_values[i] = np.mean (k_values[i - signal_period + 1:i + 1])  # 计算 %K 线的信号线（SMA）

    return k_values.tolist (), d_values.tolist ()


rsi = calculate_rsi("原始数据/train_data_15min.csv", "Close", period=14)
macd_line,  _, _ = calculate_macd("原始数据/train_data_15min.csv", "Close", short_period=7, long_period=14, signal_period=5)
k,d = calculate_stochastic_oscillator("原始数据/train_data_15min.csv", period=14)
add_column("原始数据/train_data_15min.csv", "RSI", default_value=rsi, output_file="处理过的数据/train_data_15min_modified.csv")
add_column("处理过的数据/train_data_15min_modified.csv", "MACD", default_value=macd_line)
add_column("处理过的数据/train_data_15min_modified.csv", "Stoch K", default_value=k)
add_column("处理过的数据/train_data_15min_modified.csv", "Stoch D", default_value=d)
# remove_column("原始数据/test_data_1min_original.csv", "RSI")

# concatenate_csv("train_data_15min_54321.csv","train_data_15min_0.csv","train_data_15min_54320.csv")
