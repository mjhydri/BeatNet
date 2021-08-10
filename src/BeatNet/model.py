import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
# from ConvLSTM import ConvLSTM


class BDA(nn.Module):  #beat_downbeat_activation
    def __init__(self, dim_in, num_cells, num_layers, device):
        super(BDA, self).__init__()

        self.dim_in = dim_in
        self.dim_hd = num_cells
        self.num_layers = num_layers
        self.device = device

        #added
        self.conv_out=150
        self.kernelsize=10
        self.conv1 = nn.Conv1d(1, 2, self.kernelsize)
        self.linear0 = nn.Linear(2*int((self.dim_in-self.kernelsize+1)/2), self.conv_out)     #divide to 2 is for max puling filter
        # added
        #
        # self.lstm = ConvLSTM(input_dim=self.dim_in,  # self.dim_in
        #                     hidden_dim=self.dim_hd,
        #                     kernel_size=(3, 3),
        #                     num_layers=self.num_layers,
        #                     batch_first=True,
        #                     )
        # #

        self.lstm = nn.LSTM(input_size=self.conv_out,  # self.dim_in
                            hidden_size=self.dim_hd,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=False,
                            )

        self.linear = nn.Linear(in_features=self.dim_hd,
                                out_features=3)

        self.softmax = nn.Softmax(dim=0)

        self.change_device()

    def change_device(self, device=None):
        """
        Change the device and load the model onto the new device.

        Parameters
        ----------
        device : string or None, optional (default None)
          Device to load model onto
        """

        if device is None:
            # If the function is called without a device, use the current device
            device = self.device

        # Create the appropriate device object
        device = torch.device(f'cuda:{device}'
                              if torch.cuda.is_available() else 'cpu')

        # Change device field
        self.device = device
        # Load the transcription model onto the device
        self.to(self.device)

    def forward(self, data):
        x = data
        # x = x.unsqueeze(0).unsqueeze(0).transpose(0,3).transpose(1,4).transpose(2,3)
        # x= torch.from_numpy(data)
        # x = data.float()   # I cast this to float (because torch tensors are float by default)

        #added
        x = torch.reshape(x, (-1, self.dim_in))
        x = x.unsqueeze(0).transpose(0, 1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = x.view(-1, self.num_flat_features(x))

        # 2 below belong to CNN with linear
        x = self.linear0(x)
        x = torch.reshape(x, (np.shape(data)[0], np.shape(data)[1], self.conv_out))

        # below belong to CNN without linear
        # x = torch.reshape(x, (np.shape(data)[0], np.shape(data)[1], 270))

        # x = x.unsqueeze(0).transpose(0,1)
        # added




        x = self.lstm(x)[0]
        # if self.training:
        #     x = self.lstm(x)[0]
        # else:
        #     # TODO - Need to fix up this function
        #     # Process the features in chunks
        #     batch_size = data.size(0)
        #     seq_length = data.size(1)
        #
        #     assert self.dim_in == data.size(2)
        #
        #     h = torch.zeros(self.num_layers, batch_size, self.dim_hd).to(data.device)   #batch first only affects input and output, not hidden state arrangment
        #     c = torch.zeros(self.num_layers, batch_size, self.dim_hd).to(data.device)
        #     output = torch.zeros(batch_size, seq_length, self.dim_hd).to(data.device)
        #
        #     # Forward
        #     slices = range(0, seq_length, self.inf_len)
        #     for start in slices:
        #         end = start + self.inf_len
        #         output[:, start : end, :], (h, c) = self.lstm(data[:, start : end, :], (h, c))
        #
        #     # Backward
        #     h.zero_()
        #     c.zero_()
        #
        #     for start in reversed(slices):
        #         end = start + self.inf_len
        #         result, (h, c) = self.lstm(data[:, start : end, :], (h, c))
        #         output[:, start : end, self.dim_hd:] = result[:, :, self.dim_hd:]
        #     # TODO - this is F$%#ED for now

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