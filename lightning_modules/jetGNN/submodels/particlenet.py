from torch_geometric.nn import global_mean_pool
from .edgeconv import DynamicEdgeConv

from torch import nn
import torch.nn.functional as F
import torch

from ..utils import make_mlp
from ..jet_gnn_base import JetGNNBase

    
class ParticleNet(JetGNNBase):
    """
    A vanilla GCN that simply convolves over a fully-connected graph
    """


    def __init__(self, hparams):
        super().__init__(hparams)
        
        if "spatial_channels" in hparams and hparams["spatial_channels"] is not None:
            self.spatial_channels = hparams["spatial_channels"]
        else:
            self.spatial_channels = len(self.hparams["feature_set"])
            
        # Encode input features to hidden features
        self.feature_encoder = make_mlp(
            self.spatial_channels,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # The edge networks computes new edge features from connected nodes
        self.edge_networks = [
                make_mlp(
                2 * hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_node_layer"],
                hidden_activation=hparams["hidden_activation"],
                output_activation=hparams["output_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
            ) for _ in range(hparams["n_graph_iters"])
        ]
        self.edge_convs = nn.ModuleList([
            DynamicEdgeConv(edge_network, k=hparams["knn"], aggr="add")
            for edge_network in self.edge_networks
        ])

        self.output_network = make_mlp(
            hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"] + [1],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"]
        )


    def forward(self, batch):
                
        x = self.concat_feature_set(batch).float()
        hidden_features = self.feature_encoder(x)
        
        for edge_conv in self.edge_convs:
            
            hidden_features = edge_conv(hidden_features, batch=batch.batch)

        global_average = global_mean_pool(hidden_features, batch.batch)       
        
        return self.output_network(global_average)
        