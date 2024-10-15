prompts=('domain' 'avg_degree' 'cc' 'tri' 'squ' 'max_deg' 'domain_avg_degree' 'domain_cc' 'domain_tri' 'domain_squ' 'domain_max_deg' 'avg_degree_cc' 'avg_degree_tri' 'avg_degree_squ' 'avg_degree_max_deg' 'cc_tri' 'cc_squ' 'cc_max_deg' 'tri_squ' 'tri_max_deg' 'squ_max_deg' 'avg_degree_cc_tri' 'avg_degree_cc_d_max' 'cc_tri_d_max' 'avg_degree_tri_d_max' 'avg_degree_cc_tri_d_max' 'avg_degree_cc_tri_d_max_squ')
datasets=('MUTAG' 'PROTEINS' 'NCI1' 'DD' 'ENZYMES' 'Cora' 'Citeseer' 'Pubmed' 'IMDB-BINARY' 'REDDIT-BINARY')
domains=('molecule' 'molecule' 'molecule' 'molecule' 'molecule' 'citation' 'citation' 'citation' 'social' 'social')

# Example of iterating through the list or assigning a specific value to a variable
for i in "${!datasets[@]}"; do
    for prompt in "${prompts[@]}"; do
        dataset="${datasets[$i]}"
        domain="${domains[$i]}"

        # echo "Processing prompt: $prompt, Dataset: $dataset, Domain: $domain"
        python3 main.py --dataset=$dataset --prompt=$prompt --domain=$domain > ./prompt/$dataset/$prompt.txt
        python3 eval.py --dataset=$dataset --prompt=$prompt
    done
done