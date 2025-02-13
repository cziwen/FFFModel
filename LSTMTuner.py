import json
import torch

from sklearn.metrics import mean_squared_error
from LSTMStockPredictor import LSTMStockPredictor
from Util import print_progress_bar


class LSTMTuner:
    """
    通过网格搜索找到相对较优的超参数组合，并返回最佳模型。
    """

    def __init__ (
            self,
            search_space=None,
            quick_epochs=20,
            final_epochs=100,
            input_size=1,
            output_size=1,
            config=None
    ):
        """
        :param search_space: dict, e.g. {
            "hidden_size": [16, 32],
            "num_layers": [1, 2],
            "learning_rate": [1e-3, 1e-2]
        }
        :param quick_epochs: 网格搜索时，为了节省时间的快速训练轮数
        :param final_epochs: 找到最佳组合后，将再次在训练集做充分训练的轮数
        :param input_size: LSTM 模型输入维度，若只有收盘价特征则为 1
        """
        # 若未提供搜索空间，给一个默认示例
        if search_space is None:
            search_space = {
                "hidden_size": [16, 32],
                "num_layers": [1, 2],
                "learning_rate": [1e-3, 1e-2]
            }

        self.search_space = search_space
        self.quick_epochs = quick_epochs
        self.final_epochs = final_epochs
        self.input_size = input_size
        self.output_size = output_size


        self.config = config

        # 训练搜索时存储最优结果
        self.best_model = None
        self.best_params = None
        self.best_val_mse = float ("inf")

    def _train_and_validate_model (self, X_train, y_train, X_val, y_val,
                                   hidden_size, num_layers, learning_rate,
                                   num_epochs):
        """
        训练并在验证集上计算 MSE，返回验证集 MSE 和模型。
        """
        # 1) 实例化模型
        model = LSTMStockPredictor (
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            config=self.config
        )

        # 2) 训练：这里简单训练 num_epochs 轮
        model.train_model (X_train, y_train, num_epochs=num_epochs)

        # 3) 验证集预测
        model.eval ()
        with torch.no_grad ():
            y_val_pred = model (X_val)

        # 4) 计算验证集 MSE
        val_mse = mean_squared_error (
            y_val.numpy ().flatten (),
            y_val_pred.numpy ().flatten ()
        )
        return val_mse, model

    def fit (self, X_train, y_train, X_val, y_val):
        """
        执行网格搜索，对搜索空间中的每组超参数组合训练并评估验证集 MSE。
        """
        # 取出各个超参数候选值
        hidden_sizes = self.search_space.get ("hidden_size", [32])
        num_layers_list = self.search_space.get ("num_layers", [2])
        lr_list = self.search_space.get ("learning_rate", [1e-3])

        # 逐一组合
        for hs in hidden_sizes:
            for nl in num_layers_list:
                for lr in lr_list:
                    val_mse, tmp_model = self._train_and_validate_model (
                        X_train, y_train, X_val, y_val,
                        hidden_size=hs,
                        num_layers=nl,
                        learning_rate=lr,
                        num_epochs=self.quick_epochs
                    )

                    msg = f"[GRID SEARCH] hidden_size={hs}, num_layers={nl}, lr={lr}, val_MSE={val_mse:.6f}"
                    print_progress_bar (100, msg)

                    # 若更优，则更新
                    if val_mse < self.best_val_mse:
                        self.best_val_mse = val_mse
                        self.best_params = (hs, nl, lr)
                        self.best_model = tmp_model

        print ("\n")
        print ("==" * 30)
        print (f"最佳超参数: hidden_size={self.best_params[0]}, "
               f"num_layers={self.best_params[1]}, "
               f"learning_rate={self.best_params[2]}")
        print (f"验证集最优 MSE: {self.best_val_mse:.6f}")

        # 返回该对象本身，可进行链式调用
        return self

    def train_final_model (self, X_train, y_train, save_after=False):
        """
        使用搜索得到的最佳超参数，用更多训练轮数对训练集做充分训练。
        训练完成后，更新 self.best_model。
        """
        if self.best_params is None:
            raise ValueError ("请先调用 fit() 进行超参数搜索。")

        hs, nl, lr = self.best_params
        final_model = LSTMStockPredictor (
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=hs,
            num_layers=nl,
            learning_rate=lr,
            config=self.config
        )

        final_model.train_model (X_train, y_train, self.final_epochs, save_after)

        self.best_model = final_model
        return self.best_model

    def get_best_model (self):
        """
        返回搜索得到的最佳模型（或经充分训练后的模型）。
        """
        if self.best_model is None:
            raise ValueError ("尚未进行搜索或训练，无法返回模型。")
        return self.best_model

    def get_best_params (self):
        """
        返回最佳的超参数组合。
        """
        return f"\nHidden Size: {self.best_params[0]}\nNumber of Layers: {self.best_params[1]}\nLearning Rate: {self.best_params[2]}"

    def save_config_to_json (self, file_path="CONFIG.json"):
        """
        保存 LSTM 训练配置至 JSON 文件
        """
        config_data = {
            "_comment": "此文件用于 LSTM 股票预测模型的配置，请修改参数至你需要即可。",
            "model_path": "训练好的模型/lstm_stock_model.pth",
            "ticker": self.config["ticker"],
            "start_date": self.config["start_date"],
            "end_date": self.config["end_date"],
            "seq_length": self.config["seq_length"],
            "test_ratio": self.config["test_ratio"],
            "num_epochs": self.final_epochs,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.best_params[0] if self.best_params else 100,
            "num_layers": self.best_params[1] if self.best_params else 2,
            "learning_rate": self.best_params[2] if self.best_params else 0.01
        }
        with open (file_path, "w", encoding="utf-8") as f:
            json.dump (config_data, f, ensure_ascii=False, indent=4)
        print (f"配置已保存至 {file_path}")
