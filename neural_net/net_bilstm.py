import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTMEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Двунаправленные LSTM слои
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Инициализация скрытого состояния LSTM слоёв
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)

        # Применение BiLSTM слоя
        out, _ = self.bilstm(x, (h0, c0))

        return out

class BiLSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTMDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Двунаправленные LSTM слои
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Полносвязный слой для предсказания
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Инициализация скрытого состояния LSTM слоёв
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)

        # Применение BiLSTM слоя
        out, _ = self.bilstm(x, (h0, c0))

        # Применение полносвязного слоя для предсказания
        out = self.fc(out)

        return out

class BiLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTMAutoencoder, self).__init__()

        self.encoder = BiLSTMEncoder(input_dim, hidden_dim, num_layers)
        self.decoder = BiLSTMDecoder(hidden_dim * 2, hidden_dim, num_layers, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.mean(dim=2)

        return decoded