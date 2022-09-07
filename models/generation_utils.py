# System imports
import sys
import os

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
    input_dir = "/global/cfs/cdirs/m3443/data/TrackLRP/toy_dataset_v1",
    datatype_split=[500, 100, 10], **kwargs):

    subdirs = ["train", "val", "test"]

    dataset = [load_toy_dataset(os.path.join(input_dir, subdir), num_events, **kwargs) for num_events, subdir in zip(datatype_split, subdirs)]

    return dataset

def load_toy_dataset(input_dir, num_events, **kwargs):

    # List num_events of files in input_dir
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    random.shuffle(files)
    files = files[:num_events]

    dataset = [ torch.load(f) for f in files ]
    
    return dataset

def build_dataset(
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

def generate_single_track(i, min_r, max_r, num_layers, detector_width):

    r = np.random.uniform(min_r, max_r)
    theta = np.random.uniform(0, np.pi)
    sign = np.random.choice([-1, 1])

    x = np.linspace(0.05, detector_width + 0.05, num = num_layers)
    y = sign*(np.sqrt(r**2 - (x - r*np.cos(theta))**2) - r*np.sin(theta))
    pid = np.array(len(x)*[i+1], dtype = np.int64)
    pt = 1000 * np.array(len(x)*[r])
    
    mask = (y == y)
    x, y, pid, pt = x[mask], y[mask], pid[mask], pt[mask]

    return np.vstack([x, y, pid, pt]).T

def define_truth_graph(node_feature, ptcut):

    connections = (node_feature[:-1, 2] == node_feature[1:,2])
    idxs = np.arange(len(node_feature))

    truth_graph = np.vstack([idxs[:-1][connections], idxs[1:][connections]])
    signal_truth_graph = truth_graph[:, (node_feature[:, 3][truth_graph] > ptcut).all(0)]

    return truth_graph, signal_truth_graph

def apply_geometric_cut(fully_connected_graph, node_feature, num_layers, min_r, max_r, detector_width, cut_policy = 1):

    if cut_policy >= 1:

        del_x = (node_feature[fully_connected_graph[1], 0] - node_feature[fully_connected_graph[0], 0])
        del_y = np.abs(node_feature[fully_connected_graph[1], 1] - node_feature[fully_connected_graph[0], 1])
        sine = np.sin(np.abs(np.arctan(node_feature[fully_connected_graph[1], 1]/node_feature[fully_connected_graph[1], 0]) - 
                            np.arctan(node_feature[fully_connected_graph[0], 1]/node_feature[fully_connected_graph[0], 0]))
        )
        a = np.sqrt(del_x**2 + del_y**2)
        R = a/sine/2

        fully_connected_graph = fully_connected_graph[:, (del_x <= 2*detector_width/num_layers) & (del_x > 0)]

    if cut_policy >= 2:
        R = R[(del_x <= 2*detector_width/num_layers) & (del_x > 0)]
        fully_connected_graph = fully_connected_graph[:, (R >= min_r) & (R <= max_r)]

    if cut_policy >= 3:
        R = R[(R >= min_r) & (R <= max_r)]
        del_y = np.abs(node_feature[fully_connected_graph[1], 1] - node_feature[fully_connected_graph[0], 1])
        fully_connected_graph = fully_connected_graph[:, del_y/R < 1/num_layers]
   
    fully_connected_graph = fully_connected_graph[:, node_feature[fully_connected_graph[0], 2] != node_feature[fully_connected_graph[1], 2]]

    return fully_connected_graph

def construct_training_graph(node_feature, num_layers, min_r, max_r, detector_width):

    idxs = np.arange(len(node_feature))
    fully_connected_graph = np.vstack([np.resize(idxs, (len(idxs),len(idxs))).flatten(), np.resize(idxs, (len(idxs),len(idxs))).T.flatten()])
    fully_connected_graph = fully_connected_graph[:, np.random.choice(fully_connected_graph.shape[1], size = min(1000, len(node_feature))*len(node_feature), replace = False)]

    fully_connected_graph = apply_geometric_cut(fully_connected_graph, node_feature, num_layers, min_r, max_r, detector_width)   

    return fully_connected_graph

def sample_true_fake(fully_connected_graph, signal_true_graph, eff, pur):

    truth_graph_samples = signal_true_graph[:, np.random.choice(signal_true_graph.shape[1], replace = False, size = int(eff*signal_true_graph.shape[1]))]
    if int((1-pur)/pur*truth_graph_samples.shape[1]*eff) < fully_connected_graph.shape[1]:
        fake_graph_samples = fully_connected_graph[:, np.random.choice(fully_connected_graph.shape[1], size = int((1-pur)/pur*truth_graph_samples.shape[1]*eff), replace = False)]
    else:
        fake_graph_samples = fully_connected_graph

    graph = np.concatenate([truth_graph_samples, fake_graph_samples], axis = 1)

    return graph

def apply_nhits_min(event):

    _, inverse, counts = event.pid.unique(return_inverse = True, return_counts = True)
    event.nhits = counts[inverse]
    event.pid[(event.nhits <= 3)] = 0
    event.signal_true_edges = event.signal_true_edges[:, (event.nhits[event.signal_true_edges] > 3).all(0)]

    return event

@ignore_warning(RuntimeWarning)
def generate_toy_event(num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, eff, pur):
    # pT is defined as 1000r
    
    tracks = []
    num_tracks = random.randint(num_tracks-track_dis_width, num_tracks+track_dis_width)

    # Generate all the tracks
    for i in range(num_tracks):
        track = generate_single_track(i, min_r, max_r, num_layers, detector_width)
        tracks.append(track)
    
    # Stack together track features
    node_feature = np.concatenate(tracks, axis = 0)
    
    # Define truth and training graphs
    truth_graph, signal_true_graph = define_truth_graph(node_feature, ptcut) 
    fully_connected_graph = construct_training_graph(node_feature, num_layers, min_r, max_r, detector_width)
    graph = sample_true_fake(fully_connected_graph, signal_true_graph, eff, pur)   
    
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

    event = apply_nhits_min(event)    
    
    return event

def generate_toy_dataset(num_events, num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, eff, pur):
    dataset = []
    for i in range(num_events):
        dataset.append(generate_toy_event(num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, eff, pur))
    return dataset