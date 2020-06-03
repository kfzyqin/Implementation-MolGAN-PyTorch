import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, in_features, units, activation, dropout_rate=0.):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.units = units

        self.linear1 = nn.Linear(in_features, self.units)
        self.linear2 = nn.Linear(in_features, self.units)

        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        adjacency_tensor, hidden_tensor, node_tensor = inputs
        adj = adjacency_tensor[:, :, :, 1:].permute(0, 3, 1, 2)

        annotations = torch.cat((hidden_tensor, node_tensor), -1) if hidden_tensor is not None else node_tensor

        output = torch.stack([self.linear1(annotations) for _ in range(adj.size(1))], 1)

        output = torch.matmul(adj, output)
        output = torch.sum(output, dim=1) + self.linear2(annotations)
        output = self.activation(output) if self.activation is not None else output
        output = self.dropout(output)

        return output


class MultiGraphConvolutionLayers(nn.Module):
    def __init__(self, in_features, units, activation, dropout_rate=0.):
        super(MultiGraphConvolutionLayers).__init__()
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.conv_nets = nn.ModuleList()
        for a_unit in self.units:
            self.conv_nets.append(GraphConvolution(in_features, a_unit, activation, dropout_rate))

    def forward(self, inputs):
        adjacency_tensor, hidden_tensor, node_tensor = inputs
        conv_idx = 0
        for _ in self.units:
            conv_inputs = (adjacency_tensor, hidden_tensor, node_tensor)
            hidden_tensor = self.conv_nets[conv_idx](conv_inputs)
        return hidden_tensor


class GraphAggregation(nn.Module):
    def __init__(self, in_features, out_features, b_dim, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i,j), 1)
        output = activation(output) if activation is not None\
                 else output
        output = self.dropout(output)

        return output
