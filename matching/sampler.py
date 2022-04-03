import re
import torch
from typing import Iterator, Sequence
import random
import utils
import networkx as nx
import dgl

class NMSampler(object):
    indices: Sequence[int]

    def __init__(self, graphs, min_size=5, max_size=15, seed=None, filter_negs=False,
                 sample_method="tree-pair",node_anchored=True) -> None:
        self.graphs = graphs
        self.seed = seed
        self.min_size, self.max_size, self.seed = min_size, max_size, seed
        self.filter_negs, self.sample_method = filter_negs, sample_method
        self.filter_negs = True

        self.node_anchored = node_anchored


    def sample(self, seeds):
        if self.sample_method == "tree-pair":
            size = random.randint(self.min_size + 1, self.max_size)
            graph, a = utils.sample_neigh(self.graphs, size)
            b = a[:random.randint(self.min_size, len(a) - 1)]
        elif self.sample_method == "subgraph-tree":
            graph = None
            while graph is None or len(graph) < self.min_size + 1:
                graph = random.choice(self.graphs)
            a = graph.nodes
            _, b = utils.sample_neigh([graph], random.randint(self.min_size,
                                                              graph.number_of_nodes() - 1))

        neigh_a, neigh_b = dgl.node_subgraph(graph,a), dgl.node_subgraph(graph,b)
        if self.node_anchored:
            # 修改成默认第一个节点是起始节点
            neigh_a.ndata["anchor"] = torch.Tensor([0]*neigh_a.number_of_nodes())
            neigh_b.ndata["anchor"] = torch.Tensor([0]*neigh_b.number_of_nodes())
        filter_negs = True
        while filter_negs:
            if self.sample_method == "tree-pair":
                size = random.randint(self.min_size+1, self.max_size)
                graph_a, a = utils.sample_neigh(self.graphs, size)
                graph_b, b = utils.sample_neigh(self.graphs, random.randint(self.min_size,
                    size - 1))
            elif self.sample_method == "subgraph-tree":
                graph_a = None
                while graph_a is None or len(graph_a) < self.min_size + 1:
                    graph_a = random.choice(self.graphs)
                a = graph_a.nodes
                graph_b, b = utils.sample_neigh(self.graphs, random.randint(self.min_size,
                    len(graph_a) - 1))

            neg_neigh_a, neg_neigh_b = dgl.node_subgraph(graph_a,a), dgl.node_subgraph(graph_b,b)
            if self.node_anchored:
                # 修改成默认第一个节点是起始节点
                neg_neigh_a.ndata["anchor"] = torch.Tensor([0]*neg_neigh_a.number_of_nodes())
                neg_neigh_b.ndata["anchor"] = torch.Tensor([0]*neg_neigh_b.number_of_nodes())


            matcher = nx.algorithms.isomorphism.GraphMatcher(neg_neigh_a.to_networkx(), neg_neigh_b.to_networkx())
            if not matcher.subgraph_is_isomorphic(): # a <= b (b is subgraph of a)
                filter_negs = False
                break
        # Todo 继续加油
        return neigh_a, neigh_b, neg_neigh_a, neg_neigh_a


class NMGraphCollater():
    def __init__(self, train, min_size=5, max_size=15, seed=None):
        self.graph_collate_err_msg_format = (
            "graph_collate: batch must contain DGLGraph, tensors, numpy arrays, "
            "numbers, dicts or lists; found {}")
        self.np_str_obj_array_pattern = re.compile(r'[SaUO]')
        self.train = train
        self.min_size, self.max_size, self.seed = min_size, max_size, seed

    # This implementation is based on torch.utils.data._utils.collate.default_collate
    def collate(self, items):
        """This function is similar to ``torch.utils.data._utils.collate.default_collate``.
        It combines the sampled graphs and corresponding graph-level data
        into a batched graph and tensors.

        Parameters
        ----------
        items : list of data points or tuples
            Elements in the list are expected to have the same length.
            Each sub-element will be batched as a batched graph, or a
            batched tensor correspondingly.

        Returns
        -------
        A tuple of the batching results.
        """
        print(items)
        return items
