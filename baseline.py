from llm import prompt_gen
from tqdm import tqdm
import argparse
from data import load_data
import threading
import queue
from utils import cal_graph_property  
import os
import pickle as pkl
import networkx as nx
import numpy as np
import math
from eval import eval

# Function to execute the command
def exec_command(command, globals_dict, result_queue):
    try:
        exec(command, globals_dict)
        result_queue.put("Command executed successfully")
    except Exception as e:
        result_queue.put(f"Command failed: {e}")

# Function to execute a command with a timeout
def execute_with_timeout(command, globals_dict, timeout=5):
    result_queue = queue.Queue()
    exec_thread = threading.Thread(target=exec_command, args=(command, globals_dict, result_queue))
    
    exec_thread.start()
    exec_thread.join(timeout)

    if exec_thread.is_alive():
        print("Execution timed out. Terminating the command.")
        return "Execution timed out."
    else:
        return result_queue.get()



def main(data, args):
    i = 0
    for G in tqdm(data):
        G_cc, G_num_nodes, G_num_edges, G_avg_degree, G_triangles, G_squares, G_max_deg = cal_graph_property(G)

        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

        if args.baseline == 'BA':
            model = nx.barabasi_albert_graph
        elif args.baseline == 'config':
            model = nx.configuration_model
        elif args.baseline == 'Small-world':
            model = nx.watts_strogatz_graph
        elif args.baseline == 'ER':
            model = nx.erdos_renyi_graph
        elif args.baseline == 'scale_free':
            model = nx.scale_free_graph

        #check whether a certain folder exists:
        if not os.path.exists(f'./generated_graph/{args.dataset}/{args.baseline}'):
            os.makedirs(f'./generated_graph/{args.dataset}/{args.baseline}')

        # print(G_num_nodes, G_num_edges)
        if args.baseline == 'BA': 
            p = np.arange(1, max(int(G_avg_degree)*2, 1), 1)
            best = math.inf

            for _ in p:
                gen_graph = model(G_num_nodes, _)
                diff = eval(G, gen_graph, args.dataset)
                
                current = np.mean([diff[key] for key in diff])
                if current < best:
                    best = current
                    best_m = _
            
            gen_graph = model(G_num_nodes, best_m)

        elif args.baseline == 'Small-world': #check
            p = np.arange(0, 1, 0.02)
            best = math.inf
            
            for _ in p:
                gen_graph = model(G_num_nodes, max(int(G_avg_degree), 2), _)
                diff = eval(G, gen_graph, args.dataset)
                
                current = np.mean([diff[key] for key in diff])
                if current < best:
                    best = current
                    best_p = _
            
            gen_graph = model(G_num_nodes, max(int(G_avg_degree), 2), best_p)
            
        elif args.baseline == 'ER': #check
            p = np.arange(0, 1, 0.02)
            best = math.inf

            for _ in p:
                gen_graph = model(G_num_nodes, _)
                diff = eval(G, gen_graph, args.dataset)
                
                current = np.mean([diff[key] for key in diff])
                if current < best:
                    best = current
                    best_p = _

            gen_graph = model(G_num_nodes, best_p)
        elif args.baseline == 'scale_free': #check
            alpha = np.arange(0.1, 1, 0.2)
            beta = np.arange(0.1, 1, 0.2)
            delta_in = np.arange(0.1, 1, 0.2)
            delta_out = np.arange(0.1, 1, 0.2)

            best = math.inf

            for a in alpha:
                for b in beta:
                    for d_in in delta_in:
                        for d_out in delta_out:
                            if a + b >= 1:
                                continue
                            gamma = 1 - a - b
                            # print(a, b, gamma)
                            # print(a, b, gamma, d_in, d_out)
                            gen_graph = model(G_num_nodes, alpha = a, beta = b, gamma = gamma, delta_in = d_in, delta_out = d_out)
                            gen_graph = nx.DiGraph(gen_graph).to_undirected()
                            diff = eval(G, gen_graph, args.dataset)
                            current = np.mean([diff[key] for key in diff])
                            if current < best:
                                best = current
                                best_alpha = a
                                best_beta = b
                                best_gamma = gamma
                                best_delta_in = d_in
                                best_delta_out = d_out
            
            gen_graph = model(G_num_nodes, alpha = best_alpha, beta = best_beta, gamma = best_gamma, delta_in = best_delta_in, delta_out = best_delta_out)
            # gen_graph = model(G_num_nodes, alpha = 0.41, beta = 0.54, gamma = 0.05, delta_in = 0.2, delta_out = 0.2)
            gen_graph = nx.DiGraph(gen_graph).to_undirected()
        elif args.baseline == 'config': #check
            gen_graph = model(degree_sequence, create_using = nx.Graph())
        
        pkl.dump(gen_graph, open(f'./generated_graph/{args.dataset}/{args.baseline}/graph_{i}.pkl', 'wb'))
        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--domain', type=str, default='Chemical Network')
    parser.add_argument('--baseline', type=str, default='BA')

    args = parser.parse_args()

    dataset = load_data(args.dataset)
    main(dataset, args)
