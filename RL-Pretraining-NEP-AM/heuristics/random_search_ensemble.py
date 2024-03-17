import heapq
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


class EnsembleRandomSearch:
    def __init__(self, models, seed=None):
        """
        :param models: A List of AttentionModels.
        """
        super(EnsembleRandomSearch, self).__init__()
        self.models = models
        self.seed = seed

    def _stochastic_rollout(self, env: NetworkEnhancementEnv, model: AttentionModel):
        """
        :param env:
        :param model: AttentionModel
        :return: ([node_1, node_2, ...], log prob of decoding these nodes).
        """
        # otherwise, env will be modified in-place by env.step
        env = deepcopy(env)

        with torch.no_grad():
            # action_all_steps[i] : Action in the i-th step.
            action_all_steps = []
            # log_p_all_steps[i] : log(p(action_all_steps[i]))
            log_p_a_all_steps = []
            while not env.is_terminal():
                # log_p : (batch_size, graph_size)
                # selected : (batch_size, )
                log_p, selected, _ = model.act(env)
                actions = [u if u != -1 else None for u in selected.tolist()]
                env.step(actions)
                action_all_steps.extend(actions)
                # (batch_size, 1)
                log_p_a = log_p.gather(dim=1, index=selected.unsqueeze(-1))
                log_p_a_all_steps.append(log_p_a)

            return torch.cat(log_p_a_all_steps, dim=1).sum(-1).item(), action_all_steps

    def ensemble_random_search(self, env: NetworkEnhancementEnv, n_sols):
        """
        :param env:
        :param n_sols: number of solutions to generate
        :return:
        """
        assert len(env.graphstates) == 1, 'Environment Should Contain Only One Network.'

        if self.seed is not None:
            torch.manual_seed(self.seed)

        sols = []
        for model in self.models:
            sols.extend([self._stochastic_rollout(env, model) for _ in range(n_sols)])

        # Choose top n_sols solutions
        heapq.heapify(sols)
        sols = [item[1] for item in heapq.nlargest(n_sols, sols)]

        # remove duplicate solutions
        sols.sort()
        return list(sol for sol, _ in itertools.groupby(sols))



if __name__ == '__main__':
    pass
