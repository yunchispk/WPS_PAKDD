from re import X
from turtle import shape
import torch
import torch.nn as nn

#Use GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.LeakyReLU(True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back

class Generator(nn.Module): #对每个时间点的真假作出判断
    def __init__(self, win_size, latent_dim, input_c, dropout=0.2):
        super(Generator, self).__init__()
        self.win_size = win_size
        self.n_feats = input_c
        self.n_hidden = self.n_feats//2+1
        self.n = self.n_feats# * self.win_size
        self.norm1 = nn.LayerNorm(self.n)
        self.norm2 = nn.LayerNorm(self.n)
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            ConvLayer(input_c, 3)
        )

        self.discriminator = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            # nn.Linear(self.n_hidden, self.win_size), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        ) 

    def forward(self, d): #(b,1,n)
            validity = self.discriminator(d)#(b,1,n)#.contiguous().view(validity.shape[0],-1))#(b,w,n)
            return validity#(b,1,n).view(validity.shape[0],*(self.win_size, self.n_feats)) #(b,w)

class Discriminator(nn.Module): #对每个时间点的真假作出判断
    def __init__(self, win_size, input_c, dropout=0.2):
        super(Discriminator, self).__init__()
        self.win_size = win_size
        self.n_feats = input_c
        self.n_hidden = self.n_feats//2+1
        self.n = self.n_feats# * self.win_size
        self.norm2 = nn.LayerNorm(self.n)
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            ConvLayer(input_c, 3)
        )
        self.discriminator = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        ) 

    def forward(self, d): #(b,1,n)
            validity = self.conv(d)
            validity = self.discriminator(validity)#(b,1,n)#.contiguous().view(validity.shape[0],-1))#(b,w,n)
            return validity#(b,1,n).view(validity.shape[0],*(self.win_size, self.n_feats)) #(b,w)

## LSTM_AD Model
class LSTM_AD(nn.Module):
    def __init__(self, feats):
        super(LSTM_AD, self).__init__()
        self.name = 'LSTM_AD'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 64
        self.lstm = nn.LSTM(6*feats, self.n_hidden)
        self.lstm2 = nn.LSTM(6*feats, self.n_feats)
        self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

    def forward(self, x):
        hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float32).to(device), torch.randn(1, 1, self.n_hidden, dtype=torch.float32).to(device))
        hidden2 = (torch.rand(1, 1, self.n_feats, dtype=torch.float32).to(device), torch.randn(1, 1, self.n_feats, dtype=torch.float32).to(device))
        outputs = []
        for i, g in enumerate(x):
            out, hidden = self.lstm(g.view(1, 1, -1), hidden)
            out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
            out = self.fcn(out.view(-1))
            outputs.append(1 * out.view(-1))
        v = torch.stack(outputs)
        v = v.view(v.shape[0],*(1, self.n_feats))#(0-1)
        return v#torch.stack(outputs)
