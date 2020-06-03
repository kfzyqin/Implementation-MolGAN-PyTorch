import torch.nn as nn


class EncoderRGCN(nn.Module):
    def __init__(self, units, dropout_rate=0.):
        self.graph_convolution_units, self.auxiliary_units = units


    def forward(self):
        pass