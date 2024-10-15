from llm import prompt_gen, prompt_gen2
from tqdm import tqdm
import argparse
from data import load_data
import threading
import queue
from utils import cal_graph_property  
import os

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

        if args.prompt == 'domain':
            goal = {'Domain': args.domain, 'Number_of_Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'avg_degree':
            goal = {'Average Degree': round(G_avg_degree, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'cc':
            goal = {'Clustering of Coefficient': round(G_cc, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'tri':
            goal = {'Number of Triangles': round(G_triangles, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'squ':
            goal = {'Number of Squares': round(G_squares, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'max_deg':
            goal = {'Maximum Degree': round(G_max_deg, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'domain_avg_degree':
            goal = {'Domain': args.domain, 'Average Degree': round(G_avg_degree, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'domain_cc':
            goal = {'Domain': args.domain, 'Clustering of Coefficient': round(G_cc, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'domain_tri':
            goal = {'Domain': args.domain, 'Number of Triangles': round(G_triangles, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'domain_squ':
            goal = {'Domain': args.domain, 'Number of Squares': round(G_squares, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'domain_max_deg':
            goal = {'Domain': args.domain, 'Maximum Degree': round(G_max_deg, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'avg_degree_cc':
            goal = {'Average Degree': round(G_avg_degree, 2), 'Clustering of Coefficient': round(G_cc, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'avg_degree_tri':
            goal = {'Average Degree': round(G_avg_degree, 2), 'Number of Triangles': round(G_triangles, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'avg_degree_squ':
            goal = {'Average Degree': round(G_avg_degree, 2), 'Number of Squares': round(G_squares, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'avg_degree_max_deg':
            goal = {'Average Degree': round(G_avg_degree, 2), 'Maximum Degree': round(G_max_deg, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'cc_tri':
            goal = {'Clustering of Coefficient': round(G_cc, 2), 'Number of Triangles': round(G_triangles, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'cc_squ':
            goal = {'Clustering of Coefficient': round(G_cc, 2), 'Number of Squares': round(G_squares, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'cc_max_deg':
            goal = {'Clustering of Coefficient': round(G_cc, 2), 'Maximum Degree': round(G_max_deg, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'tri_squ':
            goal = {'Number of Triangles': round(G_triangles, 2), 'Number of Squares': round(G_squares, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'tri_max_deg':
            goal = {'Number of Triangles': round(G_triangles, 2), 'Maximum Degree': round(G_max_deg, 2), 'Number of Nodes': round(G_num_nodes, 2)}
        elif args.prompt == 'squ_max_deg':
            goal = {'Number of Squares': round(G_squares, 2), 'Maximum Degree': round(G_max_deg, 2), 'Number of Nodes': round(G_num_nodes, 2)}
            
        elif args.prompt == 'avg_degree_cc_tri':
            goal = {'Average Degree': round(G_avg_degree, 2), 'Clustering of Coefficient': round(G_cc, 2), 'Number of Nodes': round(G_num_nodes, 2), 'Number of Triangles': round(G_triangles, 2)}
        elif args.prompt == 'avg_degree_cc_d_max':
            goal = {'Average Degree': round(G_avg_degree, 2), 'Clustering of Coefficient': round(G_cc, 2), 'Number of Nodes': round(G_num_nodes, 2), 'Maximum Degree': round(G_max_deg, 2)}
        elif args.prompt == 'cc_tri_d_max':
            goal = {'Clustering of Coefficient': round(G_cc, 2), 'Number of Triangles': round(G_triangles, 2), 'Number of Nodes': round(G_num_nodes, 2), 'Maximum Degree': round(G_max_deg, 2)}
        elif args.prompt == 'avg_degree_tri_d_max':
            goal = {'Average Degree': round(G_avg_degree, 2), 'Number of Triangles': round(G_triangles, 2), 'Number of Nodes': round(G_num_nodes, 2), 'Maximum Degree': round(G_max_deg, 2)}
        elif args.prompt == 'avg_degree_cc_tri_d_max':
            goal = {'Average Degree': round(G_avg_degree, 2), 'Clustering of Coefficient': round(G_cc, 2), 'Number of Nodes': round(G_num_nodes, 2), 'Number of Triangles': round(G_triangles, 2), 'Maximum Degree': round(G_max_deg, 2)}
        elif args.prompt == 'avg_degree_cc_tri_d_max_squ':
            goal = {'Average Degree': round(G_avg_degree, 2), 'Clustering of Coefficient': round(G_cc, 2), 'Number of Nodes': round(G_num_nodes, 2), 'Number of Triangles': round(G_triangles, 2), 'Maximum Degree': round(G_max_deg, 2), 'Number of Squares': round(G_squares, 2)}

        
        #check whether a certain folder exists:
        if not os.path.exists(f'../Text2Graph_round/generated_graph/{args.dataset}/{args.prompt}'):
            os.makedirs(f'../Text2Graph_round/generated_graph/{args.dataset}/{args.prompt}')


        command = prompt_gen2(i, goal, args.dataset, args.prompt).strip('`').replace('python\n', '')
        #exec command
        exec(command, globals())
        # result = execute_with_timeout(command, globals(), timeout=100)

        i += 1
        # if result == "Execution timed out.":
        #     continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--domain', type=str, default='Chemical Network')
    parser.add_argument('--prompt', type=str, default='all wo squ tri max_deg')

    args = parser.parse_args()

    dataset = load_data(args.dataset)
    main(dataset, args)
