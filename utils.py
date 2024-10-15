import pickle as pkl
import os
import networkx as nx
from itertools import combinations
from torch_geometric.datasets import MoleculeNet, Planetoid, TUDataset
from torch_geometric.utils import to_networkx

import networkx as nx
import numpy as np
import torch

# def count_squares(G):
#     squares = 0
#     for node in G.nodes():
#         neighbors = list(G.neighbors(node))
#         for i in range(len(neighbors)):
#             for j in range(i + 1, len(neighbors)):
#                 common_neighbors = set(G.neighbors(neighbors[i])).intersection(set(G.neighbors(neighbors[j])))
#                 squares += len(common_neighbors)
#     return squares // 4


# def count_squares(G):
#     # Step 1: Get the adjacency matrix of the graph
#     A = nx.to_numpy_array(G)
    
#     # Step 2: Calculate the trace of A^4 (Tr(A^4))
#     A4 = np.linalg.matrix_power(A, 4)
#     trace_A4 = np.trace(A4)
    
#     # Step 3: Calculate the degree of each node
#     degrees = np.sum(A, axis=1)
    
#     # Step 4: Calculate the sum of squares of degrees
#     sum_deg_squared = np.sum(degrees ** 2)
    
#     # Step 5: Calculate the sum of degrees
#     sum_deg = np.sum(degrees)
    
#     # Step 6: Apply the formula
#     squares_count = (trace_A4 - 2 * sum_deg_squared + sum_deg) / 8
    
#     return squares_count



def count_squares(G):
    #networkx to adj_matrix
    adj_matrix = nx.to_numpy_array(G)
    adj_matrix = torch.tensor(adj_matrix)

    k1_matrix = adj_matrix.float()
    d = adj_matrix.sum(dim=-1)
    k2_matrix = k1_matrix @ adj_matrix.float()
    k3_matrix = k2_matrix @ adj_matrix.float()
    k4_matrix = k3_matrix @ adj_matrix.float()

    diag_a4 = torch.diagonal(k4_matrix, 0)
    c4 = diag_a4 - d * (d - 1) - (adj_matrix @ d.unsqueeze(-1)).sum(dim=-1)

    return ((torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()).item()


def count_triangles(G):
    #networkx to adj_matrix
    adj_matrix = nx.to_numpy_array(G)
    adj_matrix = torch.tensor(adj_matrix)

    k1_matrix = adj_matrix.float()
    d = adj_matrix.sum(dim=-1)
    k2_matrix = k1_matrix @ adj_matrix.float()
    k3_matrix = k2_matrix @ adj_matrix.float()
    c3 = torch.diagonal(k3_matrix, 0)

    return (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float().item()


def cal_graph_property(G):
    G_cc = nx.average_clustering(G)
    G_num_nodes = G.number_of_nodes()
    G_num_edges = G.number_of_edges()

    G_avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
    G_triangles = count_triangles(G)
    G_squares = count_squares(G)
    G_max_deg = max(dict(G.degree()).values())

    return G_cc, G_num_nodes, G_num_edges, G_avg_degree, G_triangles, G_squares, G_max_deg


if __name__ == '__main__':
    dataset = TUDataset(root='./data', name='PROTEINS')
    
    data = [to_networkx(_, to_undirected = True) for _ in dataset]
    print(count_squares(data[0]))

    # print(cal_graph_property(data[-1]))