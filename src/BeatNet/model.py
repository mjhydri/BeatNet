# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>


import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class BDA(nn.Module):  #beat_downbeat_activation
    def __init__(self, dim_in, num_cells, num_layers, device):
        super(BDA, self).__init__()

        self.dim_in = dim_in
        self.dim_hd = num_cells
        self.num_layers = num_layers
        self.device = device
        self.conv_out = 150
        self.kernelsize = 10
        self.conv1 = nn.Conv1d(1, 2, self.kernelsize)
        self.linear0 = nn.Linear(2*int((self.dim_in-self.kernelsize+1)/2), self.conv_out)     #divide to 2 is for max pooling filter
        self.lstm = nn.LSTM(input_size=self.conv_out,  # self.dim_in
                            hidden_size=self.dim_hd,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=False,
                            )

        self.linear = nn.Linear(in_features=self.dim_hd,
                                out_features=3)

        self.softmax = nn.Softmax(dim=0)
        # Initialize the hidden state and cell state
        self.hidden = torch.zeros(2, 1, self.dim_hd).to(device)
        self.cell = torch.zeros(2, 1, self.dim_hd).to(device)

        self.to(device)

    def forward(self, data):
        x = data
        x = torch.reshape(x, (-1, self.dim_in))
        x = x.unsqueeze(0).transpose(0, 1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.linear0(x)
        x = torch.reshape(x, (np.shape(data)[0], np.shape(data)[1], self.conv_out))
        x, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        # x = self.lstm(x)[0]
        out = self.linear(x)
        out = out.transpose(1, 2)
        return out

    def final_pred(self, input):
        return self.softmax(input)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
