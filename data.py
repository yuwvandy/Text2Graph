import pickle as pkl
from torch_geometric.datasets import MoleculeNet, Planetoid, TUDataset
from torch_geometric.utils import to_networkx




def load_data(domain):
    if domain in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data', name=domain)
    elif domain in ['ClinTox', 'HIV', 'SIDER', 'Tox21', 'ToxCast']:
        dataset = MoleculeNet(root='./data', name=domain)
    elif domain in ['PROTEINS', 'MUTAG', 'NCI1', 'DD', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB', 'ENZYMES']:
        dataset = TUDataset(root='./data', name=domain)
    
    data = [to_networkx(_, to_undirected = True) for _ in dataset][:50]

    return data