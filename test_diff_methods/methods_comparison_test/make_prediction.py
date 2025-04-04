import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from datetime import date, timedelta
import json

from exponenta import get_stat
from get_data_all_sources import get_data

# TODO: убрать в yaml
WEIGHTS_DIR = 'data/'
NUM_DAYS_TO_PREDICT = 14
# параметры BiLSTM
INPUT_DIM = 2
HIDDEN_DIM = 128
NUM_LAYERS = 1
# параметры GRU  
INPUT_SIZE = 2
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1

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

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.GRU(input_size, hidden_size)
        self.decoder = nn.GRU(output_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq,num_step=30):
        _ , hidden_state = self.encoder(input_seq)
        
        batch_size = input_seq.size(1)
        decoder_input = torch.zeros(1, batch_size, 1) # Начальное значение декодера
        
        outputs = []
        for i in range(num_step ): #target_sequence.size(0)
            decoder_output, hidden_state = self.decoder(decoder_input, hidden_state)
            output = self.linear(decoder_output)
            outputs.append(output)
            decoder_input = output
            # Используем учителя для обучения                
        
        outputs = torch.cat(outputs, dim=0)
        return outputs

def find_trend(data):
    """
    data: np.array of shape (1, price.size, 2)
    """
    model = BiLSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    model.load_state_dict(torch.load(WEIGHTS_DIR + 'trend_bilstm', map_location='cpu'))
    
    trend = model(data)
    trend = trend.detach().numpy().squeeze()
    trend = trend/trend[-1]
    return trend

def predict_demand(data, price, b):
    model = Seq2Seq(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load(WEIGHTS_DIR + 'trend_gru', map_location='cpu'));
    predicted_demand = model(data, num_step=NUM_DAYS_TO_PREDICT).detach().numpy().squeeze(2)
    
    s = price[-1]
    price_range = price.max() - price.min()
    one_per = price_range/100
    t = np.arange(-25, 25)
    new_price = np.round(t*one_per+s).astype(int)

    norm_new_price = (new_price - price.min())/price_range
    out_grid = np.zeros((NUM_DAYS_TO_PREDICT, new_price.size))
    for i in range(NUM_DAYS_TO_PREDICT):
        out_grid[i,:] = np.exp(b[0]+b[1]*norm_new_price)*predicted_demand[i]
    return out_grid, new_price

def add_dates(df):
    """ добавляет даты, выводит формат, запрошенный Индипой"""
    dates = pd.date_range(date.today() + timedelta(days=1), periods=14)
    df.index = dates.strftime('%Y-%m-%d')
    df.columns = df.columns.astype(int)
    s = df.to_json(orient='index')
    json_dict = json.loads(s)
    res_list = list()
    for dt in json_dict:
        temp_dict = json_dict[dt]
        temp_list = list()
        for price, demand in temp_dict.items():
            temp_list.append({'price':price, 'ordersForecast':demand})
        res_list.append({'date':dt, 'prices':temp_list})
    return res_list

def run_algorithm(sku):
    df = get_data(sku)
    order = df.sales.values
    price = df.price.values
    price_norm = (price - price.min())/(price.max()-price.min())
    # extract trend
    data = torch.zeros(1, price_norm.size, 2, dtype=torch.float32)
    data[:,:,0] = torch.tensor(price_norm)
    data[:,:,1] = torch.tensor(order)
    trend = find_trend(data)
    discount_order = order/trend
    # exponenta model; returns vector of shape (5, -1) or (2, 1)
    b = get_stat(price_norm, discount_order)
    data_perm = data.permute([1,0,2])
    res_grid, new_price = predict_demand(data_perm, price, b)

    res_df = pd.DataFrame(res_grid)
    res_df.columns = list(new_price)
    res_df = res_df.round(2)
    # dates
    res_list = add_dates(res_df)
    return res_list