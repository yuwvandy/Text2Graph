import os
from openai import AzureOpenAI

parameters = {}
parameters['azure'] = {}
parameters['azure']['api_version'] = 'your version'


parameters['azure']['api_key'] = 'your key'
parameters['azure']['azure_endpoint'] = 'your endpoint'
parameters['azure']['model'] = 'your model'



client = AzureOpenAI(
        api_key=parameters['azure']['api_key'],
        azure_endpoint = parameters['azure']['azure_endpoint'],
        api_version = parameters['azure']['api_version'],
    )


def prompt_gen(i, goal, name, prompt_type):
    llm_system_prompt = 'You are a network generator who is using the Python package NetworkX to write Python code to generate a network with the user-specified property.'
    
    start = "Please write a code to generate an undirected network, with the following [PROPERTEINS]:\n"
    mid = ''.join([f'[{key}]: {goal[key]}\n' for key in goal])
    end = f'''
    
Note that :
* do not use an Erdos Renyi or Gnm graph
* make sure to generate a graph that closely matches ALL above [PROPERTIES]
* if the target property is not specified, do not modify the graph based on that property
* do not initialize the graph with no edges
* select the initial graph generator based on the domain and values of the properties requested
* different property modification should be done at the same time in one loop
* we should compare the target property and the generated property and decide the corresponding strategy

* If [Average Degree] is specified, if it is lower than target , add edges, if higher than target, remove edges
* If [Maximum Degree] is specified, if it is lower than target , add edges to the current max_degree node, if higher than target, remove edges to the current max_degree node
* If [Clustering Coefficient] is specified, if it is lower than target, randomly select a node and add edges within that node's neighbors; if higher than target, randomly select a node and remove edges within that node's neighbors
* if [Number of Triangles] is specified, if it is lower than target, randomly select a node and two of that node neighbors, and add edge in-between; if higher than target, randomly select a node and remove edges within that node's neighbors
* If [Number of Squares] is specified, randomly select a node and two of that node's neighbors; if lower than target, randomly select another node to connect with those two neighbors; if higher than target, find the intersection between neighbors of those two neighbors and remove edges to break the square
    For counting squares, using following code
        def count_squares(G):
            # Step 1: Get the adjacency matrix of the graph
            A = nx.to_numpy_array(G)
            
            # Step 2: Calculate the trace of A^4 (Tr(A^4))
            A4 = np.linalg.matrix_power(A, 4)
            trace_A4 = np.trace(A4)
            
            # Step 3: Calculate the degree of each node
            degrees = np.sum(A, axis=1)
            
            # Step 4: Calculate the sum of squares of degrees
            sum_deg_squared = np.sum(degrees ** 2)
            
            # Step 5: Calculate the sum of degrees
            sum_deg = np.sum(degrees)
            
            # Step 6: Apply the formula
            squares_count = (trace_A4 - 2 * sum_deg_squared + sum_deg) / 8
            
            return squares_count

* do not include any explanations in the outputs
* make sure code is executable
* setup the maximum iteration to 10000
* setup the threshold of relative difference to be 1%

Please only generate a python script with no additional information in the following format:
[Some function]
G = Function()
pkl.dump(G, open(f'./generated_graph/{name}/{prompt_type}/graph_{i}.pkl', 'wb'))
    '''

    llm_user_prompt = start + mid + end

    messages = [{"role": "system", "content": llm_system_prompt},
                {"role": "user", "content": llm_user_prompt}]


    
    chat_completion = client.chat.completions.create(messages=messages, model=parameters['azure']['model'],
                                                    temperature=0, seed=576879897)
    print(llm_user_prompt, flush = True)
    print(chat_completion.choices[0].message.content, flush = True)
    return chat_completion.choices[0].message.content




def prompt_gen2(i, goal, name, prompt_type):
    llm_system_prompt = 'You are a network generator who is using the Python package NetworkX to write Python code to generate a network with the user-specified property.'
    
    start = "Please write a code to generate an undirected network, with the following [PROPERTEINS]:\n"
    mid = ''.join([f'[{key}]: {goal[key]}\n' for key in goal])
    end = f'''
    
Note that :
* do not use an Erdos Renyi or Gnm graph
* make sure to generate a graph that closely matches ALL above [PROPERTIES]
* if the target property is not specified, do not modify the graph based on that property
* do not initialize the graph with no edges
* select the initial graph generator based on the domain and values of the properties requested
* do not include any explanations in the outputs
* make sure code is executable
* setup the maximum iteration to 10000
* setup the threshold of relative difference to be 1%

Please only generate a python script with no additional information in the following format:
[Some function]
G = Function()
pkl.dump(G, open(f'./generated_graph/{name}/{prompt_type}/graph_{i}.pkl', 'wb'))
    '''

    llm_user_prompt = start + mid + end

    messages = [{"role": "system", "content": llm_system_prompt},
                {"role": "user", "content": llm_user_prompt}]


    
    chat_completion = client.chat.completions.create(messages=messages, model=parameters['azure']['model'],
                                                    temperature=0, seed=576879897)
    print(llm_user_prompt, flush = True)
    print(chat_completion.choices[0].message.content, flush = True)
    return chat_completion.choices[0].message.content




if __name__ == '__main__':
    output = prompt_gen(0, 'Social Network', 100, 5, 0.7).strip('`').replace('python\n', '')

    print(output)
    exec(output)