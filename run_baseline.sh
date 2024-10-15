baselines=("BA" "config" "Small-world" "ER" "scale_free")

datasets=("MUTAG" "PROTEINS" "DD" "ENZYMES" "NCI1" "IMDB-BINARY" "REDDIT-BINARY" "Cora" "Citeseer" "Pubmed")
domains=("molecule" "molecule" "molecule" "molecule" "molecule" "social" "social" "citation" "citation" "citation")


# Example of iterating through the list or assigning a specific value to a variable
for i in "${!datasets[@]}"; do
    for baseline in "${baselines[@]}"; do
        dataset="${datasets[$i]}"
        domain="${domains[$i]}"

        echo "Processing baseline: $baseline, Dataset: $dataset, Domain: $domain"
        python3 baseline.py --dataset=$dataset --baseline=$baseline --domain=$domain
        python3 eval.py --dataset=$dataset --prompt=$baseline
    done
done

