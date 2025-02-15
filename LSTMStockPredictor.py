import torch
import torch.nn as nn
import torch.optim as optim
from Util import print_progress_bar
from src.efficient_kan import KAN


class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, learning_rate=0.01, output_size=1, config = None):
        super(LSTMStockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.config = config


        # 1) 定义 LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 2) 定义全连接层，用于回归预测
        # self.fc = nn.Linear(hidden_size, output_size)
        self.kan = KAN([hidden_size, 64, output_size])

        # 3) 定义损失函数 & 优化器，并将它们赋值给 self
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        """
        前向传播：输入 x 通过 LSTM 计算隐藏状态，再通过全连接层预测
        x 形状: (batch_size, seq_length, input_size)
        """
        batch_size = x.shape[0]

        # 初始化隐藏状态和细胞状态，全为 0
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # 通过 LSTM
        out, _ = self.lstm(x, (h_0, c_0))

        # 取最后一个时间步的输出送入全连接层
        # out = self.fc(out[:, -1, :])
        out = self.kan(out[:, -1, :])
        return out

    def train_model(self, X_train, y_train, num_epochs=100, save_after = False):
        """
        训练 LSTM 模型
        :param save_after: 训练后是否保存
        :param X_train: 训练集输入 (Tensor)
        :param y_train: 训练集目标 (Tensor)
        :param num_epochs: 训练轮数
        :param save_path: 保存模型的路径
        """
        for epoch in range(num_epochs):
            self.train()               # 进入训练模式
            self.optimizer.zero_grad() # 清空梯度

            y_pred = self(X_train)    # 前向传播
            loss = self.criterion(y_pred, y_train)  # 计算损失
            loss.backward()           # 反向传播
            self.optimizer.step()     # 更新权重

            # 更新进度条
            if (epoch + 1) % (num_epochs / 100) == 0:
                print_progress_bar (progress = (epoch / num_epochs) * 100, content = f"Loss: {loss.item ():.4f}") # 进度条



        print_progress_bar (progress = 100, content = "====== 训练完成 ======") # 进度条

        # 训练完成后保存模型
        if save_after and self.config is not None:
            torch.save(self.state_dict(), self.config["model_path"])
            print(f"模型已保存至: {self.config["model_path"]}")
