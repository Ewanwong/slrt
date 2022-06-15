import torch
import torch.nn as nn


class TemporalFusion(nn.Module):
    def __init__(self, conv_k, pool_k):
        super(TemporalFusion, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=tuple(conv_k), padding='same')
        self.pooling = nn.MaxPool1d(kernel_size=pool_k, stride=pool_k, padding='same')

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.conv(x)
        x = self.pooling(x)
        return x


class CSLR(nn.Module):
    def __init__(self, spatio_dim, num_classes, hidden_dim):
        super(CSLR, self).__init__()
        self.conv2d = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.conv2d_fc = nn.Linear(1000, spatio_dim)
        self.con1d = TemporalFusion(5, 2)
        self.con1d_fc = nn.Linear(spatio_dim, num_classes)
        self.lstm = nn.LSTM(input_size=num_classes, hidden_size=hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)


fc = nn.Linear(1000, 1024)
a = torch.zeros((2, 1000))
print(a.shape)
output = fc(a)
print(output.shape)
