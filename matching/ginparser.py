"""Parser for arguments

Put all arguments in one file and group similar arguments
"""
import argparse


class Parser():

    def __init__(self, description):
        '''
           arguments parser
        '''
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()

    def _parse(self):
        # dataset
        self.parser.add_argument(
            '--dataset', type=str, default="MUTAG",
            choices=['MUTAG', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI'],
            help='name of dataset (default: MUTAG)')
        self.parser.add_argument(
            '--batch_size', type=int, default=32,
            help='batch size for training and validation (default: 32)')
        self.parser.add_argument(
            '--fold_idx', type=int, default=0,
            help='the index(<10) of fold in 10-fold validation.')
        self.parser.add_argument(
            '--filename', type=str, default="",
            help='output file')

        # device
        self.parser.add_argument(
            '--disable-cuda', action='store_true',
            help='Disable CUDA')
        self.parser.add_argument(
            '--device', type=int, default=0,
            help='which gpu device to use (default: 0)')

        # net
        self.parser.add_argument(
            '--num_layers', type=int, default=5,
            help='number of layers (default: 5)')
        self.parser.add_argument(
            '--num_mlp_layers', type=int, default=2,
            help='number of MLP layers(default: 2). 1 means linear model.')
        self.parser.add_argument(
            '--hidden_dim', type=int, default=64,
            help='number of hidden units (default: 64)')

        # graph
        self.parser.add_argument(
            '--graph_pooling_type', type=str,
            default="sum", choices=["sum", "mean", "max"],
            help='type of graph pooling: sum, mean or max')
        self.parser.add_argument(
            '--neighbor_pooling_type', type=str,
            default="sum", choices=["sum", "mean", "max"],
            help='type of neighboring pooling: sum, mean or max')
        self.parser.add_argument(
            '--learn_eps', action="store_true",
            help='learn the epsilon weighting')

        # learning
        self.parser.add_argument(
            '--seed', type=int, default=0,
            help='random seed (default: 0)')
        self.parser.add_argument(
            '--epochs', type=int, default=350,
            help='number of epochs to train (default: 350)')
        self.parser.add_argument(
            '--lr', type=float, default=0.01,
            help='learning rate (default: 0.01)')
        self.parser.add_argument(
            '--final_dropout', type=float, default=0.5,
            help='final layer dropout (default: 0.5)')

        self.parser.add_argument(
            '--margin', type=float, default=0.1,
            help='margin for loss (default: 0.1)')
        self.parser.add_argument('--dropout', type=float, default=0.5,
                                 help='Dropout rate')
        self.parser.add_argument('--conv_type', type=str, default="SAGE",
                                 help='type of convolution')
        self.parser.add_argument('--skip', type=str, default="learnable",
                                 help='"all" or "last"')
        self.parser.add_argument('--method_type', type=str, default="order",
                                 help='type of embedding')
        self.parser.add_argument('--model_path', type=str, default="./model_path",
                                 help='path to save/load model')
        self.parser.add_argument('--test', action="store_true")
        self.args = self.parser.parse_args()
