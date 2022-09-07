import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.utils.checkpoint import checkpoint

from ..gnn_base import GNNBase, LargeGNNBase
from ..graphclassifier_base import GraphLevelClassifierBase, AttentionDeficitBase
from ..utils import make_mlp


class AGNN(GraphLevelClassifierBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        self.edge_nets = nn.ModuleList(
            [make_mlp(
                hparams["hidden"] * 2,
                [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
            ) for _ in range(hparams["n_graph_iters"])]
        )

        self.node_nets = nn.ModuleList(
            [make_mlp(
                hparams["hidden"] * 2,
                [hparams["hidden"]] * hparams["nb_node_layer"],
                hidden_activation=hparams["hidden_activation"],
                output_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
            ) for _ in range(hparams["n_graph_iters"])]
        )

        self.input_network = make_mlp(
            (hparams["spatial_channels"] + hparams["cell_channels"]),
            [hparams["hidden"]] * hparams["nb_node_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        self.output_network = make_mlp(
            hparams["hidden"] * 2 + 1,
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

    def message_step(self, x, start, end, node_net, edge_net):

        x_in = x
        
        # Apply edge network
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        e = edge_net(edge_inputs)
        e = torch.sigmoid(e)

        # Apply node network
        messages = scatter_add(e * x[start], end, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([messages, x], dim=1)
        x = node_net(node_inputs) + x_in

        return x, e

    def output_step(self, x, start, end, e, batch):

        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)

        return self.output_network(classifier_inputs).squeeze(-1)

    def forward(self, x, edge_index, batch, log_attention=False):
        
        start, end = edge_index
        x = self.input_network(x)

        attention_log = []

        for edge_net, node_net in zip(self.edge_nets, self.node_nets):
            x, attention = checkpoint(self.message_step, x, start, end, node_net, edge_net)
            
            if log_attention:
                attention_log.append(attention)
                
        if log_attention:
            return self.output_step(x, start, end, attention, batch), attention_log
        else:
            return self.output_step(x, start, end, attention, batch)

class GraphLevelAGNN(AGNN):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.output_network = make_mlp(
            hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

    def output_step(self, x, start, end, e, batch):

        # graph_level_inputs = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        graph_level_inputs = global_add_pool(x, batch)

        return self.output_network(graph_level_inputs)