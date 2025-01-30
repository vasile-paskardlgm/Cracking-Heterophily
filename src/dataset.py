import torch
import scipy
import scipy.io
import pandas as pd
import gdown
from os import path
import numpy as np
import scipy.sparse

DATAPATH = path.dirname(path.abspath(__file__)) + '/data/'

dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M 
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M 
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


#######################
class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


#######################
def load_nc_dataset(dataname):
    """ Loader for NCDataset, returns NCDataset. """
    if dataname == 'pokec': ##
        dataset = load_pokec_mat()
    elif dataname == "twitch-gamer": ##
        dataset = load_twitch_gamer_dataset() 
    elif dataname == "wiki": ##
        dataset = load_wiki()
    else:
        raise ValueError('Invalid dataname')
    return dataset


#######################
def load_pokec_mat():
    """ requires pokec.mat
    """
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdown.download(id=dataset_drive_url['pokec'], \
            output=f'{DATAPATH}pokec.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes,
                     'num_classes': 2}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


#######################
def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features

def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
            output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
            output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)
    
    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes,
                     'num_classes': 2}
    dataset.label = torch.tensor(label)
    return dataset


#######################
def load_wiki():

    if not path.exists(f'{DATAPATH}wiki_features2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_features'], \
            output=f'{DATAPATH}wiki_features2M.pt', quiet=False)
    
    if not path.exists(f'{DATAPATH}wiki_edges2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_edges'], \
            output=f'{DATAPATH}wiki_edges2M.pt', quiet=False)

    if not path.exists(f'{DATAPATH}wiki_views2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_views'], \
            output=f'{DATAPATH}wiki_views2M.pt', quiet=False)


    dataset = NCDataset("wiki") 
    features = torch.load(f'{DATAPATH}wiki_features2M.pt')
    edges = torch.load(f'{DATAPATH}wiki_edges2M.pt').T
    row, col = edges
    print(f"edges shape: {edges.shape}")
    label = torch.load(f'{DATAPATH}wiki_views2M.pt') 
    num_nodes = label.shape[0]

    print(f"features shape: {features.shape[0]}")
    print(f"Label shape: {label.shape[0]}")
    dataset.graph = {"edge_index": edges, 
                     "edge_feat": None, 
                     "node_feat": features, 
                     "num_nodes": num_nodes,
                     'num_classes': 5}
    dataset.label = label 
    return dataset 
