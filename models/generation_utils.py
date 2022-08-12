# System imports
import sys

# External imports
import numpy as np

import torch
from torch_geometric.data import Data
import scipy as sp

import warnings
import random
from typing import Type
import functools

warnings.filterwarnings("ignore")
sys.path.append("../../..")
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(
    datatype_split=[100,10,10], 
    num_tracks=100, 
    track_dis_width=10, 
    num_layers=10, 
    min_r=0.1, 
    max_r=0.5, 
    detector_width=0.5, 
    ptcut=1, 
    eff=1.0, 
    pur=0.01,
    **kwargs):

    dataset = [generate_toy_dataset(num_events, num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, eff, pur) for num_events in datatype_split]

    return dataset

def ignore_warning(warning: Type[Warning]):
    """
    Ignore a given warning occurring during method execution.
    Args:
        warning (Warning): warning type to ignore.
    Returns:
        the inner function
    """

    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category= warning)
                return func(*args, **kwargs)

        return wrapper

    return inner

def graph_intersection(
    pred_graph, truth_graph
):

    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)
    del e_1
    del e_2

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.from_numpy(
        np.vstack([e_intersection.row, e_intersection.col])
    ).long()  # .to(device)
    y = torch.from_numpy(e_intersection.data > 0)  # .to(device)
    del e_intersection
    
    return new_pred_graph, y

@ignore_warning(RuntimeWarning)
def generate_toy_event(num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, eff, pur):
    # pT is defined as 1000r
    
    tracks = []
    num_tracks = random.randint(num_tracks-track_dis_width, num_tracks+track_dis_width)
    for i in range(num_tracks):
        r = np.random.uniform(min_r, max_r)
        theta = np.random.uniform(0, np.pi)
        sign = np.random.choice([-1, 1])

        x = np.linspace(0.05, detector_width + 0.05, num = num_layers)
        y = sign*(np.sqrt(r**2 - (x - r*np.cos(theta))**2) - r*np.sin(theta))
        pid = np.array(len(x)*[i+1], dtype = np.int64)
        pt = 1000 * np.array(len(x)*[r])
        
        mask = (y == y)
        x, y, pid, pt = x[mask], y[mask], pid[mask], pt[mask]
        
        tracks.append(np.vstack([x, y, pid, pt]).T)
    
    node_feature = np.concatenate(tracks, axis = 0)
    
    connections = (node_feature[:-1, 2] == node_feature[1:,2])

    idxs = np.arange(len(node_feature))

    truth_graph = np.vstack([idxs[:-1][connections], idxs[1:][connections]])
    signal_true_graph = truth_graph[:, (node_feature[:, 3][truth_graph] > ptcut).all(0)]
    
    fully_connected_graph = np.vstack([np.resize(idxs, (len(idxs),len(idxs))).flatten(), np.resize(idxs, (len(idxs),len(idxs))).T.flatten()])
    fully_connected_graph = fully_connected_graph[:, np.random.choice(fully_connected_graph.shape[1], size = min(1000, len(node_feature))*len(node_feature), replace = False)]

    # del_x = (node_feature[fully_connected_graph[1], 0] - node_feature[fully_connected_graph[0], 0])
    # del_y = np.abs(node_feature[fully_connected_graph[1], 1] - node_feature[fully_connected_graph[0], 1])
    # sine = np.sin(np.abs(np.arctan(node_feature[fully_connected_graph[1], 1]/node_feature[fully_connected_graph[1], 0]) - 
    #                     np.arctan(node_feature[fully_connected_graph[0], 1]/node_feature[fully_connected_graph[0], 0]))
    # )
    # a = np.sqrt((node_feature[fully_connected_graph[1], 1] - node_feature[fully_connected_graph[0], 1])**2 + 
    #            (node_feature[fully_connected_graph[1], 0] - node_feature[fully_connected_graph[0], 0])**2)
    # R = a/sine/2
    # fully_connected_graph = fully_connected_graph[:, (del_x <= 2*detector_width/num_layers) & (del_x > 0) & (R>min_r) & (R<max_r) & (del_y/R < 1/num_layers)]
    # R = R[(del_x <= 2*detector_width/num_layers) & (del_x > 0) & (R>min_r) & (R<max_r) & (del_y/R < 1/num_layers)]
    # R = R[node_feature[fully_connected_graph[0], 2] != node_feature[fully_connected_graph[1], 2]]
    # fully_connected_graph = fully_connected_graph[:, node_feature[fully_connected_graph[0], 2] != node_feature[fully_connected_graph[1], 2]]

    truth_graph_samples = signal_true_graph[:, np.random.choice(signal_true_graph.shape[1], replace = False, size = int(eff*signal_true_graph.shape[1]))]
    # if int((1-pur)/pur*truth_graph_samples.shape[1]*eff) < fully_connected_graph.shape[1]:
    #     fake_graph_samples = fully_connected_graph[:, np.random.choice(fully_connected_graph.shape[1], size = int((1-pur)/pur*truth_graph_samples.shape[1]*eff), replace = False)]
    # else:
    fake_graph_samples = fully_connected_graph

    graph = np.concatenate([truth_graph_samples, fake_graph_samples], axis = 1)
    
    graph, y = graph_intersection(graph, signal_true_graph)
    node_feature = torch.from_numpy(node_feature).float()
    
    y_pid = (node_feature[:,2][graph[0]] == node_feature[:,2][graph[1]])
    pid_signal = (node_feature[:,2][graph[0]] == node_feature[:,2][graph[1]]) & (node_feature[:,3][graph]).all(0)
    
    
    event = Data(x=node_feature[:,0:2],
                 edge_index= graph,
                 graph = graph,
                 modulewise_true_edges = torch.tensor(truth_graph),
                 signal_true_edges = torch.tensor(signal_true_graph),
                 y=y,
                 pt = node_feature[:,3],
                 pid = node_feature[:,2].long(),
                 y_pid = y_pid,
                 pid_signal = pid_signal,
                )
    _, inverse, counts = event.pid.unique(return_inverse = True, return_counts = True)
    event.nhits = counts[inverse]
    event.pid[(event.nhits <= 3)] = 0
    
    event.signal_true_edges = event.signal_true_edges[:, (event.nhits[event.signal_true_edges] > 3).all(0)]
    
    return event

def generate_toy_dataset(num_events, num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, eff, pur):
    dataset = []
    for i in range(num_events):
        dataset.append(generate_toy_event(num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, eff, pur))
    return dataset