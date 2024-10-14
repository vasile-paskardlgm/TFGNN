import numpy as np
import torch
import torch_geometric.utils as utils

from torch_geometric.datasets import Planetoid,HeterophilousGraphDataset

from os import path

DATAPATH = path.dirname(path.abspath(__file__)) + '/data/'

class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        A unified and scalable class of dataset.
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
        
        """

        self.name = name  # original name
        self.graph = {}
        self.label = None

    def get_idx_split(self, train_prop=.6, valid_prop=.2):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        n = self.label.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_idx = perm[:train_num]
        valid_idx = perm[train_num:train_num + valid_num]
        test_idx = perm[train_num + valid_num:]

        split_idx = {'train': train_idx.long(),
                         'valid': valid_idx.long(),
                         'test': test_idx.long()}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(dataname:str, sub_dataname:str=''):
    """ Loader for NCDataset 
        Returns NCDataset 
    """
    dataname = dataname.lower()

    if dataname in ('cora', 'citeseer', 'pubmed', 'roman-empire', 'amazon-ratings'):
        dataset = loader(dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def loader(name):
    """ Dataset instantiation
    """
    if name in ('cora', 'citeseer', 'pubmed'):
        data = Planetoid(root=f'{DATAPATH}/Planetoid',
                                name=name)
    elif name in ('roman-empire', 'amazon-ratings'):
        data = HeterophilousGraphDataset(root=f'{DATAPATH}/HeterophilousGraphDataset',
                                name=name)

    data = data[0]

    edge_index = utils.to_undirected(edge_index=data.edge_index)
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    # print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    dataset.label = label

    dataset.num_classes = len(set(np.array(dataset.label)))
    dataset.num_nodes, dataset.num_features = dataset.graph['node_feat'].shape

    return dataset