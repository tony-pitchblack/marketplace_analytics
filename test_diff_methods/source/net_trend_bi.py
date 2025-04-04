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
        out,_ = self.bilstm(x, (h0, c0))

        return out

class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        #num_layers = 1
        self.num_layers = num_layers

        # Двунаправленные LSTM слои
        #input_dim = hidden_dim*2
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True) # , bidirectional=True

        # Полносвязный слой для предсказания
        

    def forward(self, x):
        # Инициализация скрытого состояния LSTM слоёв
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device)
        # Применение BiLSTM слоя
        out, _= self.bilstm(x, (h0, c0))
        #print(out.shape,x.shape)

        # Применение полносвязного слоя для предсказания
        #out = self.fc(out)

        return out

class PredictTrendBi(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PredictTrendBi, self).__init__()

        self.encoder = BiLSTMEncoder(input_dim, hidden_dim, num_layers)
        self.decoder = LSTMDecoder(hidden_dim*2 , hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim*2 , 1)
        #self.fc = nn.Linear(hidden_dim,1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
       # print(decoded.shape)
        decoded = self.fc(decoded)
        #decoded = decoded.mean(dim=2)

        return decoded
