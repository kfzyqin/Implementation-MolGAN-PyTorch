import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder

from layers import GraphConvolution, GraphAggregation, MultiGraphConvolutionLayers, MultiDenseLayer
from util_dir.utils_nn import PositionalEncoding


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super(Generator, self).__init__()
        self.multi_dense_layer = MultiDenseLayer(z_dim, conv_dims, torch.nn.Tanh())

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class EncoderVAE(nn.Module):
    """VAE encoder sharing part."""
    def __init__(self, conv_dim, m_dim, b_dim, z_dim, with_features=False, f_dim=0, dropout_rate=0.,
                 to_use_tf=False):
        super(EncoderVAE, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1]+m_dim, aux_dim, torch.nn.Tanh(), with_features, f_dim,
                                          dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, torch.nn.Tanh(), dropout_rate=dropout_rate)
        self.emb_mean = nn.Linear(linear_dim[-1], z_dim)
        self.emb_logvar = nn.Linear(linear_dim[-1], z_dim)

        # Transformer
        self.to_use_tf = to_use_tf
        if self.to_use_tf:
            # if False:
            nhead = 8
            nlayers = 5
            trans_dropout = 0.5
            self.pos_encoder = PositionalEncoding(graph_conv_dim[-1])
            encoder_layers = TransformerEncoderLayer(graph_conv_dim[-1], nhead, dropout=trans_dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, hidden)

        if self.to_use_tf:
            h = h.permute(1, 0, 2)
            h = self.transformer_encoder(h)
            h = h.permute(1, 0, 2)

        h = self.agg_layer(h, node, hidden)
        h = self.multi_dense_layer(h)
        h_mu = self.emb_mean(h)
        h_logvar = self.emb_logvar(h)
        h = self.reparameterize(h_mu, h_logvar)
        return h, h_mu, h_logvar


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim, m_dim, b_dim, dropout,
                 to_use_tf=False):
        super(Discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1]+m_dim, aux_dim, torch.nn.Tanh())
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, torch.nn.Tanh())

        self.output_layer = nn.Linear(linear_dim[-1], 1)

        # Transformer
        self.to_use_tf = to_use_tf
        if self.to_use_tf:
            # if False:
            nhead = 8
            nlayers = 5
            trans_dropout = 0.5
            self.pos_encoder = PositionalEncoding(graph_conv_dim[-1])
            encoder_layers = TransformerEncoderLayer(graph_conv_dim[-1], nhead, dropout=trans_dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, hidden)

        if self.to_use_tf:
            h = h.permute(1, 0, 2)
            h = self.transformer_encoder(h)
            h = h.permute(1, 0, 2)

        h = self.agg_layer(h, node, hidden)
        h = self.multi_dense_layer(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h
