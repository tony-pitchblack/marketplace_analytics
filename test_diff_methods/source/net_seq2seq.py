import torch
import torch.nn as nn


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