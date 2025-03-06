import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from LSTMModel import LSTMModel
from Util import (load_config, fetch_multi_feature_data, create_sequences,
                  plot_loss_curve, print_progress_bar, plot_stock_prediction,
                  fetch_polygon_data, save_dataframe)
from polygon import RESTClient


def prepare_data (config, csv_file=None):
    """
    下载数据、归一化，并生成 LSTM 需要的序列样本。
    返回：X, y, scaler, 原始数据（DataFrame）
    """
    if csv_file is None:
        data = fetch_multi_feature_data (
            ticker=config["ticker"],
            interval=config["interval"],
            start_date=config["start_date"],
            end_date=config["end_date"],
            features=["Close", "Open", "High", "Low", "Volume"]
        )
    else:
        data = pd.read_csv (csv_file, index_col=0, parse_dates=True)
        if "Weekday" not in data.columns:
            data["Weekday"] = data.index.dayofweek

    scaler = MinMaxScaler (feature_range=(0, 1))
    data_scaled = scaler.fit_transform (data)
    X, y = create_sequences (data_scaled, config["seq_length"])
    if len (X.shape) == 2:
        X = X.reshape (X.shape[0], X.shape[1], data.shape[1])
    return X, y, scaler, data


def load_or_create_model (config, new=False, in_n_out_Size=6):
    """
    若 new 为 False 且模型文件存在，则加载模型，否则创建新模型
    """
    model = LSTMModel (
        input_size=in_n_out_Size,
        output_size=in_n_out_Size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        learning_rate=config["learning_rate"],
        config=config
    )
    if not new and os.path.exists (config["model_path"]):
        print (f"检测到已训练模型 `{config['model_path']}`，正在加载...")
        model.load_state_dict (torch.load (config["model_path"]))
        print ("模型加载完成！")
    else:
        print ("创建新模型。")
    return model


def train_mode (config, csv_file=None, batch_size=None):
    """
    训练模式：使用配置文件参数训练模型，保存并绘制损失曲线。
    """
    print ("进入训练模式...")
    X, y, scaler, _ = prepare_data (config, csv_file)
    model = load_or_create_model (config, new=False, in_n_out_Size=y.data.size (1))
    train_losses = model.train_model (X, y, num_epochs=config["num_epochs"], save_after=True, batch_size=batch_size)
    plot_loss_curve (train_losses, title="Training Loss Curve")
    print ("训练模式完成。")


def predict_mode (config, csv_file=None):
    """
    预测模式：使用数据中最后一个序列预测下一个时间间隔的价格，
    并在控制台打印预测的时间、high、low 和 close 数值。
    """
    print ("进入预测模式...")
    X, y, scaler, data = prepare_data (config, csv_file)
    model = load_or_create_model (config, new=False, in_n_out_Size=y.data.size (1))
    model.eval ()
    # 使用最后一个序列进行单步预测
    last_seq = X[-1].unsqueeze (0)  # (1, seq_length, features)
    with torch.no_grad ():
        y_pred = model (last_seq)
    y_pred = y_pred.cpu ().numpy ()
    prediction_inv = scaler.inverse_transform (y_pred)
    pred_close = prediction_inv[0, 0]
    pred_high = prediction_inv[0, 2]
    pred_low = prediction_inv[0, 3]

    # 根据数据的最后时间戳，加上一个 interval 得到预测时间
    last_timestamp = data.index[-1]
    try:
        delta = pd.Timedelta (config["interval"])
    except Exception as e:
        print ("无法解析 config['interval'] 为 timedelta:", config["interval"])
        delta = pd.Timedelta ("1d")
    predicted_time = last_timestamp + delta

    print ("预测结果：")
    print (f"  预测时间: {predicted_time}")
    print (f"  Close: {pred_close:.3f}")
    print (f"  High : {pred_high:.3f}")
    print (f"  Low  : {pred_low:.3f}")
    print ("预测模式完成。")


def evaluate_mode (config, csv_file=None):
    """
    评价模式：对数据集中每个样本预测下一步，
    并计算 high、low、close 分别的均方误差（MSE），打印结果。
    """
    print ("进入评价模式...")
    X, y, scaler, _ = prepare_data (config, csv_file)
    model = load_or_create_model (config, new=False, in_n_out_Size=y.data.size (1))
    model.eval ()
    preds = []
    with torch.no_grad ():
        for i in range (len (X)):
            pred = model (X[i].unsqueeze (0))
            preds.append (pred.cpu ().numpy ())
    preds = np.array (preds).squeeze ()
    preds_inv = scaler.inverse_transform (preds)
    y_inv = scaler.inverse_transform (y.numpy ())
    mse_close = mean_squared_error (y_inv[:, 0], preds_inv[:, 0])
    mse_high = mean_squared_error (y_inv[:, 2], preds_inv[:, 2])
    mse_low = mean_squared_error (y_inv[:, 3], preds_inv[:, 3])
    print ("评价模式均方误差 (MSE):")
    print (f"  Close: {mse_close:.3f}")
    print (f"  High : {mse_high:.3f}")
    print (f"  Low  : {mse_low:.3f}")
    plot_stock_prediction (y_inv[:, 0], preds_inv[:, 0], config)
    print ("评价模式完成。")


def tune_mode (config):
    """
    调优模式：对训练集和验证集进行分割，
    利用网格搜索（调用模型的 tune 方法）寻找最佳超参数，
    并训练出充分训练后的新模型。
    """
    print ("进入调优模式...")
    X, y, scaler, _ = prepare_data (config)
    test_ratio = config.get ("test_ratio", 0.1)
    split_index = int (len (X) * (1 - test_ratio))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    model = load_or_create_model (config, new=True, in_n_out_Size=y.data.size (1))
    search_space = {
        "hidden_size": [config["hidden_size"] // 2, config["hidden_size"], config["hidden_size"] * 2],
        "num_layers": [config["num_layers"], config["num_layers"] + 1],
        "learning_rate": [config["learning_rate"], config["learning_rate"] / 2]
    }
    tuned_model = model.tune (X_train, y_train, X_val, y_val,
                              search_space=search_space, quick_epochs=20, final_epochs=100)
    torch.save (tuned_model.state_dict (), config["model_path"])
    print ("调优模式完成。")


if __name__ == "__main__":
    config = load_config ()
    mode = input ("请选择模式 (train/predict/evaluate/tune/download data): ").strip ().lower ()

    if mode == "train":
        batch_size = input ("请输入 batch size，默认为 full batch: ").strip ()
        if batch_size == "":
            batch_size = None
        else:
            try:
                batch_size = int (batch_size)  # 尝试转换为整数
            except ValueError:
                batch_size = None  # 不是整数，则设为 None

        use_local = input ("是否使用本地 CSV 数据进行训练? (y/n): ").strip ().lower ()
        if use_local == "y":
            csv_file = input ("请输入 CSV 文件路径: ").strip ()
            train_mode (config, csv_file=csv_file, batch_size=batch_size)
        else:
            train_mode (config, batch_size=batch_size)
    elif mode == "predict":
        use_local = input ("是否使用本地 CSV 数据进行预测? (y/n): ").strip ().lower ()
        if use_local == "y":
            csv_file = input ("请输入 CSV 文件路径: ").strip ()
            predict_mode (config, csv_file=csv_file)
        else:
            predict_mode (config)
    elif mode == "evaluate":
        use_local = input ("是否使用本地 CSV 数据进行评价? (y/n): ").strip ().lower ()
        if use_local == "y":
            csv_file = input ("请输入 CSV 文件路径: ").strip ()
            evaluate_mode (config, csv_file=csv_file)
        else:
            evaluate_mode (config)
    elif mode == "tune":
        tune_mode (config)
    elif mode == "download data":
        csv_file = input ("请输入 CSV 文件路径: ").strip ()
        df = fetch_polygon_data (
            config["ticker"], config["interval"],
            config["start_date"], config["end_date"],
            api_key=config["API_KEY"]
        )
        save_dataframe (df, csv_file)
    else:
        print ("无效模式，请输入 train, predict, evaluate, tune 或 download data.")
