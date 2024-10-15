import pickle as pkl
import os
import networkx as nx
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid, TUDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
import argparse
from collections import defaultdict
from data import load_data
from utils import count_squares




def cal_diff(g_truth, g_generated):
    if g_truth != 0 or g_generated != 0:
        return abs(g_truth - g_generated) / max(g_truth, g_generated)
    else:
        return 0
    
import signal

# Define a timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def eval(g_truth, g_generated, dataset):
    gt_cc = nx.average_clustering(g_truth)
    gen_cc = nx.average_clustering(g_generated)

    gt_num_nodes = g_truth.number_of_nodes()
    gen_num_nodes = g_generated.number_of_nodes()

    gt_num_edges = g_truth.number_of_edges()
    gen_num_edges = g_generated.number_of_edges()

    gt_avg_degree = sum(dict(g_truth.degree()).values()) / len(g_truth.nodes())
    gen_avg_degree = sum(dict(g_generated.degree()).values()) / len(g_generated.nodes())

    gt_triangles = sum(nx.triangles(g_truth).values())//3
    gen_triangles = sum(nx.triangles(g_generated).values())//3

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # Timeout set to 300 seconds

    if dataset in ['REDDIT-BINARY']:
        gt_squares = 0
        gen_squares = 0
    else:
        gt_squares = count_squares(g_truth)
        gen_squares = count_squares(g_generated)

    gt_max_deg = max(dict(g_truth.degree()).values())
    gen_max_deg = max(dict(g_generated.degree()).values())

    # print(1111, gt_triangles)

    #For each property, print groundtruth and generated one
    # if dataset in ['Cora', 'Citeseer', 'Pubmed']:
    #     print('================================================================')
    #     print(f'Ground Truth Clustering Coefficient: {gt_cc}, Generated Clustering Coefficient: {gen_cc}')
    #     print(f'Ground Truth Number of Nodes: {gt_num_nodes}, Generated Number of Nodes: {gen_num_nodes}')
    #     print(f'Ground Truth Number of Edges: {gt_num_edges}, Generated Number of Edges: {gen_num_edges}')
    #     print(f'Ground Truth Average Degree: {gt_avg_degree}, Generated Average Degree: {gen_avg_degree}')
    #     print(f'Ground Truth Number of Triangles: {gt_triangles}, Generated Number of Triangles: {gen_triangles}')
    #     print(f'Ground Truth Number of Squares: {gt_squares}, Generated Number of Squares: {gen_squares}')
    #     print(f'Ground Truth Maximum Degree: {gt_max_deg}, Generated Maximum Degree: {gen_max_deg}')
    #     print('================================================================')

    # print('================================================================')
    # print('cc', gt_cc, gen_cc)
    # print('num_nodes', gt_num_nodes, gen_num_nodes)
    # print('num_edges', gt_num_edges, gen_num_edges)
    # print('avg_degree', gt_avg_degree, gen_avg_degree)
    # print('triangles', gt_triangles, gen_triangles)
    # print('squares', gt_squares, gen_squares)
    # print('================================================================')
    
    diffs = {'cc': cal_diff(gt_cc, gen_cc), 'num_nodes': cal_diff(gt_num_nodes, gen_num_nodes), 'num_edges': cal_diff(gt_num_edges, gen_num_edges), 'avg_degree': cal_diff(gt_avg_degree, gen_avg_degree), 'triangles': cal_diff(gt_triangles, gen_triangles), 'squares': cal_diff(gt_squares, gen_squares), 'max_deg': cal_diff(gt_max_deg, gen_max_deg)}
    return diffs



    

def compare_dist(g_truth, g_generated, dataset):
    """
    Compare the distribution of the degree of the nodes and the clustering coefficient
    between the ground truth and generated NetworkX graph objects.
    
    Parameters:
    g_truth (nx.Graph): Ground truth graph
    g_generated (nx.Graph): Generated graph
    """

    # Get degree and clustering coefficient distributions
    degree_truth = list(dict(g_truth.degree()).values())
    degree_generated = list(dict(g_generated.degree()).values())
    cc_truth = list(nx.clustering(g_truth).values())
    cc_generated = list(nx.clustering(g_generated).values())

    # Create subplots for degree and clustering coefficient distributions
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Create bins for degree and clustering coefficient distributions
    bins = np.linspace(0, max(max(degree_truth), max(degree_generated)), 20)
    degree_truth_hist, _ = np.histogram(degree_truth, bins=bins, density=True)
    degree_generated_hist, _ = np.histogram(degree_generated, bins=bins, density=True)
    
    bins_center = (bins[:-1] + bins[1:]) / 2

    # Plot degree distribution with dots and lines
    ax[0].plot(bins_center, degree_truth_hist, marker='o', color='blue', label='Truth')
    ax[0].plot(bins_center, degree_generated_hist, marker='o', color='orange', label='Generated')
    ax[0].set_title('Degree Distribution')
    ax[0].set_xlabel('Degree')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # Create bins for clustering coefficient distributions
    bins = np.linspace(0, 1, 20)
    cc_truth_hist, _ = np.histogram(cc_truth, bins=bins, density=True)
    cc_generated_hist, _ = np.histogram(cc_generated, bins=bins, density=True)

    bins_center = (bins[:-1] + bins[1:]) / 2

    # Plot clustering coefficient distribution with dots and lines
    ax[1].plot(bins_center, cc_truth_hist, marker='o', color='blue', label='Truth')
    ax[1].plot(bins_center, cc_generated_hist, marker='o', color='orange', label='Generated')
    ax[1].set_title('Clustering Coefficient Distribution')
    ax[1].set_xlabel('Clustering Coefficient')
    ax[1].set_ylabel('Density')
    ax[1].legend()

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.savefig(f'result/{dataset}/degree_cc_dist.png')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--prompt_type', type=str, default='all wo domain')
    args = parser.parse_args()

    dataset = load_data(args.dataset)

    diffs = defaultdict(list)

    for i, g in enumerate(dataset):
        graph_name = f'graph_{i}.pkl'
        
        if not os.path.exists(f'./generated_graph/{args.dataset}/{args.prompt_type}/{graph_name}'):
            continue
        G = pkl.load(open(f'./generated_graph/{args.dataset}/{args.prompt_type}/{graph_name}', 'rb'))

        diff = eval(g, G, args.dataset)

        for key in diff:
            diffs[key].append(diff[key])

        # if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
        #     compare_dist(g, G, args.dataset)
    
    #save the below information to a .txt file
    with open(f'result/{args.dataset}/evaluation-{args.prompt_type}.txt', 'w') as f:
        f.write(f'Evaluation for {args.dataset}\n')
        # print(cc_diffs)
        for key in diffs:
            f.write(f'Average Relative {key} Difference: {sum(diffs[key])/len(diffs[key])} +- {np.std(diffs[key])}\n')