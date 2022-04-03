import dgl
import numpy as np
import scipy.stats as stats
import random
def sample_neigh(graphs, size):
    ps = np.array([g.number_of_nodes() for g in graphs], dtype=np.float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        #graph = random.choice(graphs)
        graph = graphs[idx]
        start_node = random.choice(range(0,graph.number_of_nodes()))
        neigh = [start_node]
        #所有邻居 = 入度+出度的邻居
        in_sub,_ = graph.in_edges(neigh)
        _,out_sub = graph.out_edges(neigh)
        frontier = list(set(np.concatenate((out_sub,in_sub),axis =0)) - set(neigh))
        # pyg：frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            #new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)

            in_sub,_ = graph.in_edges([new_node])
            _,out_sub = graph.out_edges([new_node])
            frontier += list(set(np.concatenate((out_sub,in_sub),axis =0)))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh


def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
            help='Optimizer weight decay.')
    parser.add_argument('--seed', type=int, default=0,help='random seed (default: 0)')
