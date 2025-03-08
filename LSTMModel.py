import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

from Util import print_progress_bar
from src.efficient_kan import KAN

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, learning_rate=0.01, output_size=1, config=None):
        """
        初始化 LSTM 模型

        :param input_size: 输入特征维度
        :param hidden_size: 隐藏层神经元个数
        :param num_layers: LSTM 层数
        :param learning_rate: 学习率
        :param output_size: 输出维度
        :param config: 配置字典
        """
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.config = config

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 使用 KAN 结构替代传统全连接层
        self.kan = KAN([hidden_size, 64, output_size])

        # 损失函数 & 优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        """
        前向传播：x -> LSTM -> KAN 模块
        :param x: 输入张量，形状 (batch_size, seq_length, input_size)
        :return: 输出张量，形状 (batch_size, output_size)
        """
        batch_size = x.shape[0]
        # 初始化隐藏状态和细胞状态（全0）
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        # LSTM 前向计算
        out, _ = self.lstm(x, (h_0, c_0))
        # 取最后一个时间步的输出送入 KAN 模块
        out = self.kan(out[:, -1, :])
        return out

    def train_model(self, X_train, y_train, num_epochs=100, save_after=False, batch_size=None):
        """
        训练模型
        :param X_train: 训练输入张量
        :param y_train: 训练目标张量
        :param num_epochs: 训练轮数
        :param save_after: 是否在训练后保存模型（需 config 中配置 model_path）
        :param batch_size: 每个batch的尺寸
        :return: 每个 epoch 的训练损失列表
        """
        if batch_size is not None: # Batch gradient descent(mini-batch)
            dataset = TensorDataset (X_train, y_train)
            dataloader = DataLoader (dataset, batch_size=batch_size, shuffle=True)

            train_losses = []
            for epoch in range (num_epochs):
                self.train ()
                epoch_loss = 0.0

                for X_batch, y_batch in dataloader:
                    self.optimizer.zero_grad ()
                    y_pred = self (X_batch)
                    loss = self.criterion (y_pred, y_batch)
                    loss.backward ()
                    self.optimizer.step ()
                    epoch_loss += loss.item ()

                # 计算平均损失
                epoch_loss /= len (dataloader)
                train_losses.append (epoch_loss)

                current_progress = (epoch + 1) / num_epochs * 100
                print_progress_bar (progress=current_progress,
                                    content=f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.10f}")

            print ("====== 训练完成 ======")
            if save_after and self.config is not None:
                torch.save (self.state_dict (), self.config["model_path"])
                print (f"模型已保存至: {self.config['model_path']}")

            return train_losses

        else: # Full batch training
            train_losses = []
            for epoch in range(num_epochs):
                self.train()
                self.optimizer.zero_grad()
                y_pred = self(X_train)
                loss = self.criterion(y_pred, y_train)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                current_progress = (epoch + 1) / num_epochs * 100
                print_progress_bar(progress=current_progress, content=f"Loss: {loss.item():.10f}")

            print("====== 训练完成 ======")
            if save_after and self.config is not None:
                torch.save(self.state_dict(), self.config["model_path"])
                print(f"模型已保存至: {self.config['model_path']}")
            return train_losses

    def tune(self, X_train, y_train, X_val, y_val, search_space=None, quick_epochs=20, final_epochs=100):
        """
        网格搜索超参数（hidden_size, num_layers, learning_rate），
        先使用少量训练轮数快速搜索，再用最佳超参数进行充分训练。

        :param X_train: 训练集输入张量
        :param y_train: 训练集目标张量
        :param X_val: 验证集输入张量
        :param y_val: 验证集目标张量
        :param search_space: 超参数搜索空间字典，例如：
                             {"hidden_size": [50, 80],
                              "num_layers": [2, 3],
                              "learning_rate": [0.0005, 0.001]}
                             若为 None，则使用当前模型参数作为唯一候选。
        :param quick_epochs: 网格搜索时快速训练的轮数
        :param final_epochs: 使用最佳超参数后进行充分训练的轮数
        :return: 经充分训练后的最佳模型
        """
        if search_space is None:
            search_space = {
                "hidden_size": [self.hidden_size],
                "num_layers": [self.num_layers],
                "learning_rate": [self.learning_rate]
            }

        best_val_mse = float("inf")
        best_params = None

        # 遍历所有超参数组合
        for hs in search_space.get("hidden_size", [self.hidden_size]):
            for nl in search_space.get("num_layers", [self.num_layers]):
                for lr in search_space.get("learning_rate", [self.learning_rate]):
                    # 实例化临时模型
                    tmp_model = LSTMModel(
                        input_size=self.input_size,
                        output_size=self.output_size,
                        hidden_size=hs,
                        num_layers=nl,
                        learning_rate=lr,
                        config=self.config
                    )
                    # 快速训练
                    tmp_model.train_model(X_train, y_train, num_epochs=quick_epochs)
                    tmp_model.eval()
                    with torch.no_grad():
                        y_val_pred = tmp_model(X_val)
                    # 计算验证集 MSE
                    val_mse = mean_squared_error(
                        y_val.numpy().flatten(),
                        y_val_pred.numpy().flatten()
                    )
                    print_progress_bar(100, f"[GRID SEARCH] hidden_size={hs}, num_layers={nl}, lr={lr}, val_MSE={val_mse:.6f}")

                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_params = (hs, nl, lr)

        print("\n" + "==" * 30)
        print(f"最佳超参数: hidden_size={best_params[0]}, num_layers={best_params[1]}, learning_rate={best_params[2]}")
        print(f"验证集最优 MSE: {best_val_mse:.6f}")

        # 使用最佳超参数重新训练（充分训练）
        final_model = LSTMModel(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=best_params[0],
            num_layers=best_params[1],
            learning_rate=best_params[2],
            config=self.config
        )
        final_model.train_model(X_train, y_train, num_epochs=final_epochs)
        return final_model


