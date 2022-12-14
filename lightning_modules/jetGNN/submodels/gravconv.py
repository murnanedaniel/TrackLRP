from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torch
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, knn_graph, radius_graph
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..jet_gnn_base import JetGNNBase
from ..utils import make_mlp


class GravConv(nn.Module):
    def __init__(self, hparams, input_size=None, output_size=None):
        super().__init__()
        self.hparams = hparams
        self.feature_dropout = hparams["feature_dropout"] if "feature_dropout" in hparams else 0.0
        self.spatial_dropout = hparams["spatial_dropout"] if "spatial_dropout" in hparams else 0.0
        self.input_size = hparams["hidden"] if input_size is None else input_size
        self.output_size = hparams["hidden"] if output_size is None else output_size
        

        self.feature_network = make_mlp(
                2*(self.input_size + 1),
                [self.output_size] * hparams["nb_node_layer"],
                output_activation=hparams["hidden_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                dropout=self.feature_dropout
        )

        self.spatial_network = make_mlp(
                self.input_size + 1,
                [self.input_size] * hparams["nb_node_layer"] + [hparams["emb_dims"]],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                dropout=self.spatial_dropout
        )

        if self.hparams["grav_level"] > 1:
            self.mass_network = make_mlp(
                self.input_size + 1,
                [self.input_size] * hparams["nb_node_layer"] + [1],
                hidden_activation=hparams["hidden_activation"],
                output_activation= "Sigmoid",
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
            )

        if "learned_grav_weight" in hparams and hparams["learned_grav_weight"]:
            self.grav_fact_network = make_mlp(
                3*(hparams["hidden"]+1),
                [hparams["hidden"]] * hparams["nb_node_layer"] + [1],
                hidden_activation=hparams["hidden_activation"],
                output_activation= hparams["grav_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
            )

        self.setup_configuration()

    def get_neighbors(self, spatial_features):
        
        edge_index = torch.empty([2, 0], dtype=torch.int64, device=spatial_features.device)
 
        if self.use_radius:
            radius_edges = radius_graph(spatial_features, r=self.r, max_num_neighbors=self.hparams["max_knn"], batch=self.batch)
            edge_index = torch.cat([edge_index, radius_edges], dim=1)
        
        if self.use_knn and self.knn > 0:
            k_edges = knn_graph(spatial_features, k=self.knn, batch=self.batch, loop=True)
            edge_index = torch.cat([edge_index, k_edges], dim=1)

        if self.use_rand_k and self.rand_k > 0:
            random_edges = knn_graph(torch.rand(spatial_features.shape[0], 2, device=spatial_features.device), k=self.rand_k, batch=self.batch, loop=True) 
            edge_index = torch.cat([edge_index, random_edges], dim=1)
        
        # Remove duplicate edges
        edge_index = torch.unique(edge_index, dim=1)

        return edge_index

    def get_grav_function(self, hidden_features, edge_index, d):
        start, end = edge_index
        # Handle the various possible versions of "attention gravity"
        if "learned_grav_weight" in self.hparams and self.hparams["learned_grav_weight"]:
            self.global_features = torch.cat([global_add_pool(hidden_features, self.batch), global_mean_pool(hidden_features, self.batch), global_max_pool(hidden_features, self.batch)], dim=1)
            self.edge_index = edge_index
        if "grav_level" not in self.hparams or self.hparams["grav_level"] in [0, 1]:
            grav_weight = self.grav_weight
            grav_function = - grav_weight * d / self.r**2
        elif self.hparams["grav_level"] == 2:
            grav_weight = None
            m = self.mass_network(hidden_features)
            grav_function = - d / m[end].squeeze()
        elif self.hparams["grav_level"] == 3:
            grav_weight = None
            m = self.mass_network(hidden_features)
            grav_function = - d / (m[start].squeeze() * m[end].squeeze())
        
        return grav_function, grav_weight

    def get_attention_weight(self, spatial_features, hidden_features, edge_index):
        start, end = edge_index
        d = torch.sum((spatial_features[start] - spatial_features[end])**2, dim=-1) 
        grav_function, grav_fact = self.get_grav_function(hidden_features, edge_index, d)

        return torch.exp(grav_function), grav_fact

    def grav_pooling(self, spatial_features, hidden_features):
        edge_index = self.get_neighbors(spatial_features)
        start, end = edge_index
        d_weight, grav_fact = self.get_attention_weight(spatial_features, hidden_features, edge_index)

        if "grav_level" in self.hparams and self.hparams["grav_level"] == 0:
            hidden_features = F.normalize(hidden_features, p=1, dim=-1)

        return scatter_add(hidden_features[start] * d_weight.unsqueeze(1), end, dim=0, dim_size=hidden_features.shape[0]), edge_index, grav_fact

    def forward(self, hidden_features, batch, current_epoch):
        self.current_epoch = current_epoch
        self.batch = batch

        hidden_features = torch.cat([hidden_features, hidden_features.mean(dim=1).unsqueeze(-1)], dim=-1)
        spatial_features = self.spatial_network(hidden_features)

        if "norm" in self.hparams:
            spatial_features = F.normalize(spatial_features)

        aggregated_hidden, edge_index, grav_fact = self.grav_pooling(spatial_features, hidden_features)
        concatenated_hidden = torch.cat([aggregated_hidden, hidden_features], dim=-1)
        return self.feature_network(concatenated_hidden), edge_index, spatial_features, grav_fact



    def setup_configuration(self):
        self.current_epoch = 0
        self.use_radius = bool("r" in self.hparams and self.hparams["r"])
        self.use_knn = bool("knn" in self.hparams and self.hparams["knn"])
        self.use_rand_k = bool("rand_k" in self.hparams and self.hparams["rand_k"])

    @property
    def r(self):
        if isinstance(self.hparams["r"], list):
            if len(self.hparams["r"]) == 2:
                return self.hparams["r"][0] + ( (self.hparams["r"][1] - self.hparams["r"][0]) * self.current_epoch / self.hparams["max_epochs"] )
            elif len(self.hparams["r"]) == 3:
                # A function that scales linearly between the first and second value of the list for half the epochs, then scales between the second and third value for the second half of the epochs
                if self.current_epoch < self.hparams["max_epochs"]/2:
                    return self.hparams["r"][0] + ( (self.hparams["r"][1] - self.hparams["r"][0]) * self.current_epoch / (self.hparams["max_epochs"]/2) )
                else:
                    return self.hparams["r"][1] + ( (self.hparams["r"][2] - self.hparams["r"][1]) * (self.current_epoch - self.hparams["max_epochs"]/2) / (self.hparams["max_epochs"]/2) )
        elif isinstance(self.hparams["r"], float):
            return self.hparams["r"]
        else:
            return 0.3

    @property
    def knn(self):
        if isinstance(self.hparams["knn"], list):
            if len(self.hparams["knn"]) == 2:
                return int( self.hparams["knn"][0] + ( (self.hparams["knn"][1] - self.hparams["knn"][0]) * self.current_epoch / self.hparams["max_epochs"] ) )
            elif len(self.hparams["knn"]) == 3:
                # A function that scales linearly between the first and second value of the list for half the epochs, then scales between the second and third value for the second half of the epochs
                if self.current_epoch < self.hparams["max_epochs"]/2:
                    return int( self.hparams["knn"][0] + ( (self.hparams["knn"][1] - self.hparams["knn"][0]) * self.current_epoch / (self.hparams["max_epochs"]/2) ) )
                else:
                    return int( self.hparams["knn"][1] + ( (self.hparams["knn"][2] - self.hparams["knn"][1]) * (self.current_epoch - self.hparams["max_epochs"]/2) / (self.hparams["max_epochs"]/2) ) )
        else:
            return self.hparams["knn"]

    @property
    def rand_k(self):
        if isinstance(self.hparams["rand_k"], list):
            return int( self.hparams["rand_k"][0] + ( (self.hparams["rand_k"][1] - self.hparams["rand_k"][0]) * self.current_epoch / self.hparams["max_epochs"] ) )
        else:
            return self.hparams["rand_k"]

    @property
    def grav_weight(self):
        if "learned_grav_weight" in self.hparams and self.hparams["learned_grav_weight"]:
            grav_weight_multiplier = self.grav_fact_network(self.global_features)
            grav_weight_multiplier = grav_weight_multiplier[self.batch[self.edge_index[0]]].squeeze()
        else: 
            grav_weight_multiplier = 1.0
        
        if "grav_weight" not in self.hparams:
            return grav_weight_multiplier
        if isinstance(self.hparams["grav_weight"], list):
            return grav_weight_multiplier * (self.hparams["grav_weight"][0] + (self.hparams["grav_weight"][1] - self.hparams["grav_weight"][0]) * self.current_epoch / self.hparams["max_epochs"])
        elif isinstance(self.hparams["grav_weight"], float):
            return grav_weight_multiplier * self.hparams["grav_weight"]
        