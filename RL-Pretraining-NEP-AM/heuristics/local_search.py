import os
import torch
from torch.utils.data.dataloader import DataLoader

from heuristics.beam_search import BeamSearch, BeamSearchModel
from problems.network_enhancement_env import NetworkEnhancementEnv
from problems.network_generator import construct_network_seeds
from problems.network_dataset import NetworkDataset
from problems.graph_state import GraphState
from copy import copy, deepcopy
from problems.targeted_removal import TargetedRemoval
from utils.functions import load_params, get_freer_gpu
import networkx as nx
import heapq
import numpy as np


def apply(actions, env):
    """
    param actions: List. Each element is a node.
    param env:
    return:
    """
    env = deepcopy(env)
    for action in actions:
        env.step([action])
    return env


class LocalSearch(BeamSearch):
    def __init__(self,
                 model: BeamSearchModel,
                 swap_size,
                 swap_step_size,
                 beam_width,
                 nbor_size,
                 method,
                 n_mc_sims,
                 sim_seed,
                 reuse_hash):
        """
        param model:
        param swap_size: n_swap-OPT. In each swap, n_swap nodes would be swapped out.
        param swap_step_size: Interval between the swap starting points.
        param beam_width: Beam Search width.
        param nbor_size: Size of the search neighborhood.
        param method: Attack method.
        param n_mc_sims: Number of Simulations.
        """
        super(LocalSearch, self).__init__(model, beam_width)
        self.swap_size = swap_size
        self.swap_step_size = swap_step_size
        self.nbor_size = nbor_size
        assert self.swap_size % 2 != 0 or self.swap_step_size % 2 == 0, \
            'When swap an even number of nodes, swap_step_size must be even.'
        self.method = method
        self.n_mc_sims = n_mc_sims
        self.sim_seed = sim_seed
        self.reuse_hash = reuse_hash

    def _apply_actions_partially(self, actions, env: NetworkEnhancementEnv, s_idx):
        """
        Apply action not in actions[s_idx, s_idx+swap_size-1] to env.
        param actions: List. Each element is a node.
        param env:
        param s_idx:
        return:
        """
        env = deepcopy(env)

        l = len(actions)
        e_idx = (s_idx + self.swap_size - 1) % l

        idx_skipped = None
        if self.swap_size % 2 != 0 and s_idx % 2 == 0:
            # (e_idx, e_idx+1) is an edge, for e_idx is an even number
            idx_skipped = (e_idx + 1) % l
            idx_to_apply = (idx_skipped + 1) % l
        else:
            # when swap_size is an even number, we apply from e_idx + 1 to s_idx - 1
            # when swap_size is an odd number and s_idx % 2 != 0, we also apply from e_idx + 1 to s_idx - 1
            idx_to_apply = (e_idx + 1) % l

        while idx_to_apply != s_idx:
            env.step([actions[idx_to_apply]])
            idx_to_apply = (idx_to_apply + 1) % l

        if idx_skipped is not None:
            env.step([actions[idx_skipped]])
        return env

    def local_search(self, init_sol, env: NetworkEnhancementEnv, swap_size=None):
        """
        :param init_sol: List. Each element is a node.
        :param env:
        :param swap_size:
        :return:
        """

        # Important to overwrite the self.swap_size
        if swap_size is not None:
            self.swap_size = swap_size

        assert self.swap_size % 2 != 0 or self.swap_step_size % 2 == 0, \
            'When swap an even number of nodes, swap_step_size must be even.'

        init_sol = copy(init_sol)
        len_sol = len(init_sol)
        assert len_sol % 2 == 0, 'Length of the Initial Solution Should be Even.'

        s_idx = 0
        last_swap_idx = 0  # record last swapped location

        init_robustness = apply(init_sol, env).calculate_robustness(self.method,
                                                                    self.n_mc_sims,
                                                                    self.sim_seed,
                                                                    self.reuse_hash)[0]

        while True:
            # Apply actions not in actions[s_idx, s_idx+swap_size-1] to env.
            partial_interacted_env = self._apply_actions_partially(init_sol, env, s_idx)

            # Generate Neighborhood
            assert len(env.graphstates) == 1, 'Environment Should Contain Only One Network.'
            n_nodes = env.graphstates[0].graph.number_of_nodes()

            if self.swap_size == 1 and n_nodes <= self.nbor_size:
                candidates = np.where(partial_interacted_env.graphstates[0].node_banned == False)[0].tolist()
                candidates = [[_] for _ in candidates]
            else:
                candidates = self.beam_search(partial_interacted_env)
                heapq.heapify(candidates)
                candidates = [item[1] for item in heapq.nlargest(self.nbor_size, candidates)]

            opt_candidate = None
            opt_robustness = init_robustness

            for candidate in candidates:
                robustness = apply(candidate,
                                   partial_interacted_env).calculate_robustness(self.method,
                                                                                self.n_mc_sims,
                                                                                self.sim_seed,
                                                                                self.reuse_hash)[0]
                if robustness > opt_robustness:
                    opt_candidate = candidate
                    opt_robustness = robustness

            if opt_candidate is not None:
                last_swap_idx = s_idx
                """
                Let e_idx = s_idx + swap_size - 1. 
                when e_idx is an even number, (e_idx, e_idx+1) is an edge. 
                init_sol[e_idx + 1] is applied to env in the last step.
                Thus we put opt_candidate[0] at init_sol[e_idx]
                     we put opt_candidate[1] ~ opt_candidate[swap_size-1] at init_sol[s_idx] ~ init_sol[e_idx -1]            
                """
                bias = 0
                if self.swap_size % 2 != 0 and s_idx % 2 == 0:
                    init_sol[(s_idx + self.swap_size - 1) % len_sol] = opt_candidate[0]
                    bias = 1

                for i in range(len(opt_candidate) - bias):
                    init_sol[(s_idx + i) % len_sol] = opt_candidate[i + bias]

                print(f'rejoice: improve the robustness from {init_robustness:.4f} '
                      f'to {opt_robustness:.4f} ')
                init_robustness = opt_robustness

            s_idx = (s_idx + self.swap_step_size) % len_sol

            if last_swap_idx == s_idx:
                break

        # print("Finish the improvement of an initial solution.")

        return init_robustness, init_sol

    def advanced_local_search(self, init_sol, env: NetworkEnhancementEnv, swap_size_list):
        """
        Args:
            init_sol: List of nodes.
            env:
            swap_size_list: List of swap_sizes.
        Returns:
        """

        last_imp_idx = 0
        idx = 0

        while True:
            swap_size = swap_size_list[idx]
            print(f'Local Search with swap size: {swap_size} ')
            robustness, sol = self.local_search(init_sol, env, swap_size)
            if init_sol != sol:
                last_imp_idx = idx

            init_sol = sol
            idx = (idx + 1) % len(swap_size_list)
            if last_imp_idx == idx:
                print('-' * 10 + 'END' + '-' * 10)
                break

        return robustness, sol


if __name__ == '__main__':
    model_saving_dir = '../outputs/BA_n_20_m_2/run_20230928T101248'
    checkpoint = 'epoch-19.pt'

    # Read Parameters from Model Saving Dir
    opts_ = load_params(model_saving_dir)

    # Set the device
    if opts_.use_cuda:
        opts_.device = torch.device(get_freer_gpu())
    else:
        opts_.device = torch.device("cpu")

    # Instantiate the Model
    model_ = BeamSearchModel(
        opts_.node_dim,
        opts_.embedding_dim,
        opts_.feed_forward_hidden,
        opts_.n_encode_layers,
        opts_.n_heads,
        opts_.tanh_clipping,
        opts_.aggr_func,
        opts_.device
    ).to(opts_.device)
    model_.set_decode_type("greedy")
    model_.eval()

    checkpoint = torch.load(os.path.join(model_saving_dir, checkpoint))
    if 'model_state_dict' in checkpoint:
        model_.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_.load_state_dict(checkpoint)

    opts_.val_size = 128

    _, val_seeds, _ = construct_network_seeds(opts_.epoch_size, opts_.val_size, opts_.test_size)
    val_dataset = NetworkDataset(val_seeds, opts_)

    # Initialize DataLoader
    val_dataloader = DataLoader(val_dataset, 1)

    for graphs in val_dataloader:
        val_graphs = [nx.from_numpy_array(graph.numpy()) for graph in graphs]

        graphstates = [GraphState(_, opts_.edge_budget_percentage) for _ in val_graphs]

        env_ = NetworkEnhancementEnv(graphstates)

        robustness_ = env_.calculate_robustness(TargetedRemoval, 1)[0]

        print(f'Avg Robustness of Origin Env: {robustness_:.2f}')

        # Get An initial solution
        action_all_steps_ = model_.rollout(env_)
        actions_ = [list(item) for item in zip(*action_all_steps_)][0]

        robustness_1 = apply(actions_, env_).calculate_robustness(TargetedRemoval, 1)[0]
        print(f'Robustness of Initial Solution: {robustness_1:.2f}')

        local_search = LocalSearch(model_, swap_size=1, swap_step_size=1, beam_width=2, nbor_size=100,
                                   method=TargetedRemoval, n_mc_sims=1, sim_seed=2778, reuse_hash=False)

        swap_size_list = [1, 5, 7]
        _, imp_actions = local_search.advanced_local_search(actions_, env_, swap_size_list)

        # _, imp_actions = local_search.local_search(actions_, env_)
        robustness_2 = apply(imp_actions, env_).calculate_robustness(TargetedRemoval, 1)[0]

        print(f'Robustness of Final Solution: {robustness_2:.2f}')

        print('*' * 10)
