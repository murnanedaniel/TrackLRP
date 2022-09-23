import os
import numpy as np
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import itertools
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch import nn
from torch_cluster import radius_graph, knn_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import frnn
    FRNN_AVAILABLE = True
except ImportError:
    FRNN_AVAILABLE = False

def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
    batch_norm=True,
    dropout=0.0,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers with dropout
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(hidden_activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)



def open_processed_files(input_dir, num_jets):
    
    jet_files = os.listdir(input_dir)
    num_files = (num_jets // 100000) + 1
    jet_paths = [os.path.join(input_dir, file) for file in jet_files][:num_files]
    opened_files = [torch.load(file) for file in tqdm(jet_paths)]
    
    opened_files = list(itertools.chain.from_iterable(opened_files))
    
    return opened_files

def load_processed_datasets(input_dir,  data_split, graph_construction):
    
    print("Loading torch files")
    print(time.ctime())
    train_jets = open_processed_files(os.path.join(input_dir, "train"), data_split[0])
    val_jets = open_processed_files(os.path.join(input_dir, "val"), data_split[1])
    test_jets = open_processed_files(os.path.join(input_dir, "test"), data_split[2])
    
    print("Building events")
    print(time.ctime())
    train_dataset = build_processed_dataset(train_jets, graph_construction,  data_split[0])
    val_dataset = build_processed_dataset(val_jets, graph_construction, data_split[1])
    test_dataset = build_processed_dataset(test_jets, graph_construction, data_split[2])
    
    return train_dataset, val_dataset, test_dataset
    


def build_processed_dataset(jetlist, graph_construction, num_jets = None):
    
    subsample = jetlist[:num_jets] if num_jets is not None else jetlist

    try:
        _ = subsample[0].px
    except Exception:
        for i, data in enumerate(subsample):
            subsample[i] = Data.from_dict(data.__dict__)

    if (graph_construction == "fully_connected"):        
        for jet in subsample:
            jet.edge_index = get_fully_connected_edges(jet.x)

    print("Testing sample quality")
    for sample in tqdm(subsample):
        sample.x = sample.px

        # Check if any nan values in sample
        for key in sample.keys:
            assert not torch.isnan(sample[key]).any(), "Nan value found in sample"
            
    return subsample

"""
Returns an array of edge links corresponding to a fully-connected graph - NEW VERSION
"""
def get_fully_connected_edges(x):
    
    n_nodes = len(x)
    node_list = torch.arange(n_nodes)
    edges = torch.combinations(node_list, r=2).T
    
    return torch.cat([edges, edges.flip(0)], axis=1)

def build_edges(
    query, database, indices=None, r_max=1.0, k_max=10, return_indices=False, remove_self_loops=True
):

    dists, idxs, _, _ = frnn.frnn_grid_points(
        points1=query.unsqueeze(0),
        points2=database.unsqueeze(0),
        lengths1=None,
        lengths2=None,
        K=k_max,
        r=r_max,
        grid=None,
        return_nn=False,
        return_sorted=True,
    )

    idxs = idxs.squeeze().int()
    ind = torch.Tensor.repeat(
        torch.arange(idxs.shape[0], device=device), (idxs.shape[1], 1), 1
    ).T.int()
    positive_idxs = idxs >= 0
    edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()

    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    if remove_self_loops:
        edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return (edge_list, dists, idxs, ind) if return_indices else edge_list

def find_neighbors(embedding1, embedding2, r_max=1.0, k_max=10):
    embedding1 = embedding1.clone().detach().reshape((1, embedding1.shape[0], embedding1.shape[1]))
    embedding2 = embedding2.clone().detach().reshape((1, embedding2.shape[0], embedding2.shape[1]))
    
    dists, idxs, _, _ = frnn.frnn_grid_points(points1 = embedding1,
                                          points2 = embedding2,
                                          lengths1 = None,
                                          lengths2 = None,
                                          K = k_max,
                                          r = r_max,
                                         )
    return idxs.squeeze(0)

def FRNN_graph(embeddings, r, k):
    
    idxs = find_neighbors(embeddings, embeddings, r_max=r, k_max=k)

    positive_idxs = (idxs.squeeze() >= 0)
    ind = torch.arange(idxs.shape[0], device = positive_idxs.device).unsqueeze(1).expand(idxs.shape)
    edges = torch.stack([ind[positive_idxs],
                        idxs[positive_idxs]
                        ], dim = 0)     
    return edges