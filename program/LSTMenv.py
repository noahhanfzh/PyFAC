import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import os
import csv


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


def trainLSTM(working_library, device):
    seq_len = 10
    hidden_dim = 512
    num_layers = 3
    learning_rate = 0.001
    num_epochs = 2000
    batch_size = 64

    inputs, outputs = read_training_data(working_library, seq_len)

    input_dim = inputs.shape[2]
    output_dim = outputs.shape[1]

    inputs = torch.from_numpy(np.array(inputs)).to(device)
    outputs = torch.from_numpy(np.array(outputs)).to(device)

    train_dataset = torch.utils.data.TensorDataset(inputs, outputs)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

    return model, loss_history

def predictLSTM(model, input, device):
    input = torch.from_numpy(np.array(input)).to(device)
    model.eval()
    with torch.no_grad():
        output_ = model(input)
        return output_.tolist()


def read_training_data(working_library, seq_len):
    input_ = []
    output_ = []

    temp_ = []
    with open(f'{working_library}\\LSTM_input.csv', 'r') as file_:
        reader_ = csv.reader(file_)
        for row_ in reader_:
            if row_ == []:
                continue
            row_ = [float(x) for x in row_]
            temp_.append(row_)

    for i in range(len(temp_) - seq_len - 1):
        input_.append(temp_[i:i + seq_len])

    temp_ = []
    with open(f'{working_library}\\LSTM_output.csv', 'r') as file_:
        reader_ = csv.reader(file_)
        for row_ in reader_:
            if row_ == []:
                continue
            row_ = [float(x) for x in row_]
            temp_.append(row_)

    for i in range(len(temp_) - seq_len - 1):
        output_.append(temp_[i:i + seq_len])

    return np.array(input_), np.array(output_)



