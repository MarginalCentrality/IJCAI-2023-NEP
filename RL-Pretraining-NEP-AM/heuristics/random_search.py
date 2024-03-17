import os
import torch
import numpy as np
import networkx as nx
from copy import copy, deepcopy
from torch.utils.data.dataloader import DataLoader
from nets.attention_model import AttentionModel
from problems.network_enhancement_env import NetworkEnhancementEnv
from problems.network_generator import construct_network_seeds
from problems.network_dataset import NetworkDataset
from utils.functions import load_params, get_freer_gpu
import itertools


class RandomSearch:
    def __init__(self, model: AttentionModel, seed=None):
        """
        :param model:
        :param n_sols: Number of Solutions to Generate.
        """
        super(RandomSearch, self).__init__()
        self.model = model
        self.model.set_decode_type('sampling')
        self.model.eval()
        self.seed = seed

    def _stochastic_rollout(self, env: NetworkEnhancementEnv):
        """
        :param env:
        :return: paths from root to leaves, each path contains (log prob of the path, [node_1, node_2, ...]).
        """
        # otherwise, env will be modified in-place by env.step
        env = deepcopy(env)

        with torch.no_grad():
            # action_all_steps[i] : Action in the i-th step.
            action_all_steps = []

            while not env.is_terminal():
                _, selected, _ = self.model.act(env)
                actions = [u if u != -1 else None for u in selected.tolist()]
                env.step(actions)
                action_all_steps.extend(actions)

            return action_all_steps

    def random_search(self, env: NetworkEnhancementEnv, n_sols):
        """
        :param env:
        :param n_sols: number of solutions to generate
        :return:
        """
        assert len(env.graphstates) == 1, 'Environment Should Contain Only One Network.'

        if self.seed is not None:
            torch.manual_seed(self.seed)

        sols = [self._stochastic_rollout(env) for _ in range(n_sols)]

        # remove duplicate solutions
        sols.sort()
        return list(sol for sol, _ in itertools.groupby(sols))


if __name__ == '__main__':
    model_saving_dir = '../outputs/BA_n_20_m_2/run_20221127T232037'
    checkpoint = 'epoch-2.pt'
    n_sols = 6

    # Read Parameters from Model Saving Dir
    opts_ = load_params(model_saving_dir)

    # Set the device
    if opts_.use_cuda:
        opts_.device = torch.device(get_freer_gpu())
    else:
        opts_.device = torch.device("cpu")

    # Instantiate the Model
    model_ = AttentionModel(
        opts_.node_dim,
        opts_.embedding_dim,
        opts_.feed_forward_hidden,
        opts_.n_encode_layers,
        opts_.n_heads,
        opts_.tanh_clipping,
        opts_.device
    ).to(opts_.device)

    random_search = RandomSearch(model_)

    checkpoint = torch.load(os.path.join(model_saving_dir, checkpoint))
    if 'model_state_dict' in checkpoint:
        model_.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_.load_state_dict(checkpoint)

    opts_.val_size = 1

    _, val_seeds, _ = construct_network_seeds(opts_.epoch_size, opts_.val_size, opts_.test_size)
    val_dataset = NetworkDataset(val_seeds, opts_)

    # Initialize DataLoader
    val_dataloader = DataLoader(val_dataset, opts_.val_size)

    for graphs in val_dataloader:
        val_graphs = [nx.from_numpy_array(graph.numpy()) for graph in graphs]
        env_ = NetworkEnhancementEnv(val_graphs,
                                     opts_.edge_budget_percentage)

        # Sample initial solutions
        print(random_search.random_search(env_, n_sols))
