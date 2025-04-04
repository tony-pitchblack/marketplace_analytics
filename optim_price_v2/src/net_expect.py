import torch
import torch.nn as nn
class ExpectParams(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ExpectParams, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 2)  # Учитываем два направления LSTM

    def forward(self, x):
        # Проходим через LSTM
        _, (ht, _) = self.lstm(x)
        # Берем последнее состояние LSTM с двух направлений
        out = torch.cat((ht[-2], ht[-1]), dim=1)  # Объединяем последнее состояние из обоих направлений
        out = self.fc(out)  # Применяем полносвязный слой
        return out