import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from Util import *
from LSTMStockPredictor import LSTMStockPredictor
from LSTMTuner import LSTMTuner

# 1. 配置管理：集中存放常量、参数
CONFIG = load_config ()


def prepare_data (config, need_seperate=False):
    """
    准备数据：获取真实股票数据、归一化、切分训练测试集
    :param config: 参数配置字典
    :param need_seperate: 是否需要将数据分成训练集和测试集
    :return: (X_train, y_train, X_test, y_test, scaler) if need_seperate=True
             (X_Tensor, y_Tensor, scaler) if need_seperate=False
    """
    # 1. 获取真实股票数据
    stock_prices = fetch_multi_feature_data (
        ticker=config["ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        features=["Close", "Open", "High", "Low", "Volume"]
    )  # shape: (num_samples, num_features)

    # 2. 归一化
    scaler = MinMaxScaler (feature_range=(0, 1))
    stock_prices_scaled = scaler.fit_transform (stock_prices)

    # 3. 创建序列数据
    X_tensor, y_tensor = create_sequences (stock_prices_scaled, config["seq_length"])
    # 确保 X_tensor 形状为 (batch_size, seq_length, num_features)
    if len (X_tensor.shape) == 2:
        num_features = stock_prices.shape[1]  # 获取特征数
        X_tensor = X_tensor.reshape (X_tensor.shape[0], X_tensor.shape[1], num_features)

    # 4. 如果需要切分训练/测试集
    if need_seperate:
        test_size = int (len (X_tensor) * config["test_ratio"])
        train_size = len (X_tensor) - test_size
        X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
        return X_train, y_train, X_test, y_test, scaler
    else:
        return X_tensor, y_tensor, scaler


def load_or_create_model (config):
    """
    如果模型已存在，则加载模型；否则创建新的模型
    """
    model_path = config["model_path"]
    input_size = config["input_size"]
    output_size = config["output_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    learning_rate = config["learning_rate"]

    # 如果模型已存在，直接加载
    if os.path.exists (model_path):
        print (f"检测到已训练的模型 `{model_path}`，正在加载...")
        model = (LSTMStockPredictor
                 (input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate,
                  output_size=output_size, config=config))
        model.load_state_dict (torch.load (model_path))
        print ("模型加载完成，可以直接使用！")
    else:
        # 否则创建新模型
        print ("未找到训练好的模型，正在创建新模型...")
        model = (LSTMStockPredictor
                 (input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate,
                  output_size=output_size, config=config))
        print ("新模型已创建，需要训练！")
    return model


def train_and_evaluate_main (config):
    """
    主流程：准备数据 -> 加载/创建模型 -> 训练 -> 测试 -> 绘图
    """
    # ========== (1) 数据准备 ==========
    X_train, y_train, X_test, y_test, scaler = prepare_data (config, need_seperate=True)

    # ========== (2) 加载或创建模型 ==========
    model = load_or_create_model (config)

    # ========== (3) 训练模型 ==========
    model.train_model (X_train, y_train, num_epochs=config["num_epochs"], save_after=True)

    # ========== (4) 测试模型进行预测 ==========
    evaluate (model, X_test, y_test, config, scaler)


def evaluate (model, X_test, y_test, config, scaler):
    model.eval ()  # 进入评估模式
    with torch.no_grad ():
        y_test_pred = model (X_test)  # 预测
        y_test_pred = y_test_pred.cpu ().numpy ()  # 转换为 NumPy
        y_test_actual = y_test.cpu ().numpy ()  # 转换为 NumPy

    # 确保数据形状匹配，如果为1
    if y_test_actual.ndim == 1:
        y_test_actual = y_test_actual.reshape (-1, 1)  # (batch_size, 1)
    if y_test_pred.ndim == 1:
        y_test_pred = y_test_pred.reshape (-1, 1)  # (batch_size, 1)

    # 逆归一化
    y_test_pred = scaler.inverse_transform (y_test_pred)  # 预测值逆归一化
    y_test_actual = scaler.inverse_transform (y_test_actual)  # 真实值逆归一化

    # 计算误差指标（针对 `Close` 价格）
    mse = mean_squared_error (y_test_actual[:, 0], y_test_pred[:, 0])  # 只评估 Close 价格
    rmse = np.sqrt (mse)
    variance = np.var (y_test_actual[:, 0] - y_test_pred[:, 0])

    print ("\n")
    print (f"Mean Squared Error (MSE): {mse:.4f}")
    print (f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print (f"Prediction Variance: {variance:.4f}")

    # 绘制预测结果（仅展示 `Close` 价格）
    plot_stock_prediction (y_test_actual[:, 0], y_test_pred[:, 0], config["ticker"])


def evaluate_main (config):
    # ========== (1) 加载数据 (整合成 X, y 不分训练/测试) ==========
    X_tensor, y_tensor, scaler = prepare_data (config, need_seperate=False)

    # ========== (2) 加载或创建模型 ==========
    model = load_or_create_model (config)

    # ========== (3) 进行预测并评估 ==========
    evaluate (model, X_tensor, y_tensor, config, scaler)


def tune_main (config):
    """
    使用超参数搜索来找到相对最佳的 hidden_size / num_layers / learning_rate。
    这里会进行训练集/验证集切分，不触及最终的测试集。
    """

    # ========== (A) 先拿到“整体”数据 X_tensor, y_tensor ==========
    X_train, y_train, X_val, y_val, scaler = prepare_data (config, need_seperate=True)

    # ========== (C) 定义搜索空间，这里仅示例，可根据你需求定制 ==========
    search_space = {
        "hidden_size": np.arange (50, 250, 15).tolist (),  # 扩展更小和更大的隐藏层大小
        "num_layers": [2, 3, 4],  # 增加层数的选择范围
        "learning_rate": np.linspace (1e-4, 1e-1, 6)  # 细化学习率的选择
    }

    # ========== (D) 创建 tuner，进行网格搜索 ==========
    tuner = LSTMTuner (
        search_space=search_space,
        quick_epochs=20,  # 快速搜索
        final_epochs=100,  # 再训练最优组合
        input_size=config["input_size"],  # 如果你只有收盘价特征，通常是1
        output_size=config["output_size"],
        config=config
    )
    tuner.fit (X_train, y_train, X_val, y_val)

    # ========== (E) 用搜索到的最佳参数，再做充分训练 ==========
    tuner.save_config_to_json ()  # 保存 CONFIG
    best_model = tuner.train_final_model (X_train, y_train, save_after=False)

    # ========== (F) 评估一下在验证集上表现如何 ==========
    evaluate (best_model, X_val, y_val, config, scaler)


def predict_main (config, future_days):
    """
    使用已经训练好的模型，预测未来 `future_days` 个交易日的所有特征价格/数值
    """

    # ============== 1. 加载全部数据（X_tensor, y_tensor，不分训练/测试） ==============
    # 假设 X_tensor 形状: (num_samples, seq_length, num_features)
    #      y_tensor 形状: (num_samples, num_features)  或  (num_samples,)（如果只预测一个特征）
    X_tensor, y_tensor, scaler = prepare_data (config, need_seperate=False)

    if not os.path.exists (config["model_path"]):
        print ("❌ 未检测到已训练模型，请先执行 'train' 进行训练，然后再来预测。")
        return

    # ============== 2. 加载模型，切换到推理模式 ==============
    model = load_or_create_model (config)
    model.eval ()

    # ============== 3. 取最后一个序列作为预测起点 ==============
    # X_tensor[-1] 的形状: (seq_length, num_features)
    # unsqueeze(0) 后形状: (1, seq_length, num_features)，对应 batch_size=1
    last_seq = X_tensor[-1].unsqueeze (0)

    predictions = []  # 用来存储未来若干天的预测结果

    # ============== 4. 循环预测未来 days ==============
    with torch.no_grad ():
        for _ in range (future_days):
            # 4.1 模型预测
            #     假设模型输出形状: (batch_size=1, num_features)
            pred = model (last_seq)  # pred.shape -> (1, num_features)
            pred_value_scaled = pred.cpu ().numpy ()  # numpy, shape: (1, num_features)

            # 4.2 逆归一化，得到真正数值
            #     scaler.inverse_transform() 期望形状 (batch_size, num_features)
            pred_value = scaler.inverse_transform (pred_value_scaled)  # shape: (1, num_features)

            # 4.3 保存预测结果（只保存一个数组或者拆开都行）
            #     这里把每个未来时刻的多特征预测存为一行
            predictions.append (pred_value.flatten ())  # shape: (num_features,)

            # 4.4 将本次预测的“新值”拼到序列末尾，移除最前面旧值，以继续预测下一时刻
            #     注意：要拼接回“归一化”后的值，因为模型需要的是缩放后的输入
            #           我们已经有 pred_value_scaled (1, num_features) 可直接用
            new_value_scaled = torch.tensor (pred_value_scaled, dtype=torch.float32)  # (1, num_features)
            # 当前 last_seq: (1, seq_length, num_features)
            # 去掉第一步 (dim=1 的第 0 个)，然后在末尾追加新预测
            # new_value_scaled.unsqueeze(1) -> (1, 1, num_features)
            new_value_scaled = new_value_scaled.unsqueeze (1)
            last_seq = torch.cat ([last_seq[:, 1:, :], new_value_scaled], dim=1)
            # 结果 last_seq 仍然是 (1, seq_length, num_features)

    # ============== 5. 打印预测结果 ==============
    print (f"未来 {future_days} 个交易日的预测结果(多特征):")
    for i, val in enumerate (predictions, start=1):
        # val.shape -> (num_features,)
        # 如果你只想看第一个特征(比如 Close)，可以取 val[0]
        print (f"Day {i}: {val}")

    # ============== 6. 可视化(示例) ==============
    # 假设 y_tensor.shape -> (num_samples, num_features)
    all_actual = scaler.inverse_transform (y_tensor.cpu ().numpy ())  # shape: (num_samples, num_features)
    # 这里简单把最后一个真实值索引拿出来 + 未来的预测
    predictions = np.array (predictions)  # 现在 predictions.shape = (future_days, num_features)
    plot_future_prediction (all_actual[:, 0], predictions[:, 0], config["ticker"])  # 这里只取第一列 close


def plot_future_prediction (historical_data, future_preds, ticker):
    """
    简单绘图：
    - `historical_data` 是历史真实数据（长度 M）
    - `future_preds` 是未来预测的列表（长度 N）
    """

    historical_length = len (historical_data)
    future_length = len (future_preds)

    plt.figure (figsize=(10, 5))
    plt.plot (range (historical_length), historical_data, label="Historical", color="blue")

    # 未来的部分，从 historical_length 往后画
    plt.plot (range (historical_length, historical_length + future_length),
              future_preds, label="Predicted Future", color="red", marker='o')

    plt.title (f"Predict Next {future_length} Days - {ticker}")
    plt.xlabel ("Days")
    plt.ylabel ("Price")
    plt.legend ()
    plt.show ()


if __name__ == "__main__":
    action = input ("请输入操作类型 (train/evaluate/tune/predict): ").strip ().lower ()

    if action == "train":
        train_and_evaluate_main (CONFIG)
    elif action == "evaluate":
        evaluate_main (CONFIG)
    elif action == "tune":
        tune_main (CONFIG)
    elif action == "predict":
        future_days = int (input ("请输入要预测的交易日数: "))
        predict_main (CONFIG, future_days)
    else:
        print ("❌ 无效输入，请输入 'train' / 'evaluate' / 'tune' / 'predict'")
