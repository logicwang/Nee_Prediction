import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ChainedResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(ChainedResidualBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = nn.ConstantPad1d((-(padding), 0), 0) # 保证因果卷积
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = nn.ConstantPad1d((-(padding), 0), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 这里的参数需要对齐你的 Config
        num_channels = [configs.d_model] * configs.e_layers
        kernel_size = 3
        dropout = configs.dropout

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = configs.enc_in if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [ChainedResidualBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_len, Features] -> 需要转置为 [Batch, Features, Seq_len]
        y = x_enc.transpose(1, 2)
        y = self.tcn(y)
        y = self.linear(y.transpose(1, 2))
        return y[:, -48:, :] # 返回预测长度的部分