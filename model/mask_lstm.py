"""
带特征掩码机制的用于预测驾驶员意图的深度学习模型。
"""

import torch.nn as nn


# LSTM + MLP 模型较为简单，训练效果还行
class NaiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc1(last_hidden_state)
        out = self.fc2(out)
        # predictions = F.log_softmax(out, dim=1)
        return out


# 用于轨迹预测，已废弃
class TrajLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        lstm_out, _ = self.lstm(x)
        batch_size, seq_len, features = lstm_out.shape
        lstm_out = lstm_out.contiguous().view(batch_size * seq_len, features)
        predictions = self.fc(lstm_out)
        predictions = predictions.view(batch_size, seq_len, -1)
        return predictions
