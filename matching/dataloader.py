"""
PyTorch compatible dataloader
"""


import math
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import dgl
from dgl.dataloading import GraphDataLoader
from matching.sampler import NMSampler


class NMDataLoader():
    def __init__(self,
                 dataset,
                 batch_size,
                 device,
                 collate_fn=None,
                 seed=0,
                 shuffle=True,
                 split_name='fold10',
                 fold_idx=0,
                 split_ratio=0.7):

        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}

        labels = [l for _, l in dataset]
        graphs = [graph for graph, l in dataset]

        if split_name == 'fold10':
            train_idx, valid_idx = self._split_fold10(
                labels, fold_idx, seed, shuffle)
        elif split_name == 'rand':
            train_idx, valid_idx = self._split_rand(
                labels, split_ratio, seed, shuffle)
        else:
            raise NotImplementedError()

        graph_len = len(graphs)
        train_sampler = NMSampler(graphs[0:int(graph_len*0.7)])
        valid_sampler = NMSampler(graphs[int(graph_len*0.7)::])

        self.train_loader = GraphDataLoader(
            dataset, collate_fn=train_sampler.sample,
            batch_size=batch_size,  **self.kwargs)
        self.valid_loader = GraphDataLoader(
            dataset, collate_fn=valid_sampler.sample,
            batch_size=batch_size,  **self.kwargs)

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader

    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True):
        ''' 10 flod '''
        assert 0 <= fold_idx and fold_idx < 10, print(
            "fold_idx must be from 0 to 9.")

        skf = StratifiedKFold(n_splits=10, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):    # split(x, y)
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]

        print(
            "train_set : test_set = %d : %d",
            len(train_idx), len(valid_idx))

        return train_idx, valid_idx

    def _split_rand(self, labels, split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))
        np.random.seed(seed)
        np.random.shuffle(indices)
        split = int(math.floor(split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print(
            "train_set : test_set = %d : %d",
            len(train_idx), len(valid_idx))

        return train_idx, valid_idx

