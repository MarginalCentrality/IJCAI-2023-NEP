import json
import os
import pickle
import time
from copy import deepcopy

import torch
from torch.utils.data.dataloader import DataLoader

from problems.network_enhancement_env import NetworkEnhancementEnv
from problems.network_generator import construct_network_seeds
from problems.network_dataset import EnhancedNetworkDataset
from problems.graph_state import GraphState
from problems.random_removal import RandomRemoval
from problems.targeted_removal import TargetedRemoval
from nets.attention_model import AttentionModel
from policy.reinforce import REINFORCE
from utils.functions import load_params, get_freer_gpu
import argparse
import networkx as nx
from math import sqrt, ceil
from math import inf
from tabulate import tabulate
import numpy as np
import csv

from utils.choose_checkpoint import get_res_name


def get_edge_budget(num_node, edge_budget_percentage):
    edge_budget = ceil((num_node * (num_node - 1)) / 2.0 * edge_budget_percentage / 100.0)
    return edge_budget


def get_robustness(env):
    if args.n_process == 1:
        robustness_estimation = np.array(env.calculate_robustness(opts.instantiated_mtd,
                                                                  opts.n_mc_sims,
                                                                  opts.sim_seed,
                                                                  opts.reuse_hash))
    elif args.n_process > 1:
        robustness_estimation = np.array(env.calculate_robustness_parallel(opts.instantiated_mtd,
                                                                           opts.n_mc_sims,
                                                                           args.n_process,
                                                                           opts.sim_seed,
                                                                           opts.reuse_hash))

    return robustness_estimation


@torch.no_grad()
def evaluate():
    """
    :param
    :return:
    """

    # Fetch the Test Data
    # IF the graphs have been stored in disk,
    # they will be loaded directly.
    test_seeds, _, _ = construct_network_seeds(opts.test_size, 0, 0)
    test_dataset = EnhancedNetworkDataset(test_seeds, opts)

    assert len(test_dataset) == opts.test_size, "The test dataset size and test batch size are unequal~"

    # Initialize DataLoader
    test_dataloader = DataLoader(test_dataset, opts.test_size)

    # Load the test env
    for graphs, seeds in test_dataloader:
        # numpy array ----> networkx object
        test_graphs = [nx.from_numpy_array(graph.numpy()) for graph in graphs]

        # Construct graph states
        graphstates = [GraphState(g, opts.edge_budget_percentage) for g in test_graphs]
        env = NetworkEnhancementEnv(graphstates)

    seeds = [_.item() for _ in seeds]

    init_robustness_estimation = get_robustness(env)

    # Load the modifications
    # {checkpoint : {seed: [ u, v ] }}
    modifications = None
    with open(precomputing_res_path, 'rb') as file:
        modifications = pickle.load(file)

    names, test_imps = [], []
    # Evaluate each checkpoint
    for checkpoint in modifications:
        # Apply the modifications
        applied_env = deepcopy(env)
        for i in range(n_step):
            actions = [modifications[checkpoint][seed][i] for seed in seeds]
            applied_env.step(actions)

        robustness_estimation = get_robustness(applied_env)
        names.append(checkpoint)
        # (test_batch_size, )
        test_imps.append(robustness_estimation - init_robustness_estimation)

    # (n_checkpoints, test_batch_size)
    test_imps = np.stack(test_imps, axis=0)
    # (test_batch_size, )
    best_imp = test_imps.max(axis=0)

    # Compute the contribution of each checkpoint
    # (n_checkpoints, )
    contribs = (test_imps == best_imp).sum(axis=-1)

    # (n_checkpoints, )
    min_test_imps = test_imps.min(axis=1)
    avg_test_imps = test_imps.mean(axis=1)
    max_test_imps = test_imps.max(axis=1)

    content = ""
    for name, min_test_imp, avg_test_imp, max_test_imp, contrib in zip(names, min_test_imps,
                                                                       avg_test_imps, max_test_imps, contribs):
        content = content + name + "\t" + f'min_imp:{min_test_imp:.4f}' \
                  + "\t" + f'avg_imp:{avg_test_imp:.4f}' \
                  + "\t" + f'max_imp:{max_test_imp:.4f}' \
                  + "\t" + f'contrib:{contrib}' + "\n"

    content = content + "Ensemble Model" + "\t" \
              + f'min_imp:{best_imp.min():.4f}' + "\t" \
              + f'avg_imp:{best_imp.mean():.4f}' + "\t" \
              + f'max_imp:{best_imp.max():.4f}'
    return content


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate Models"
    )

    parser.add_argument('--model_saving_dir', type=str, default='./', help="The Directory of Saving Models.")
    parser.add_argument('--graph_model', type=str, default='BA_n_20_m_2', help="The graph model (out-of-distribution) "
                                                                               "to evaluate")
    parser.add_argument('--edge_budget_percentage', type=float, default='1.0')
    parser.add_argument('--largest_possible_ebp', type=float, default='1.0')

    parser.add_argument('--test_size', type=int, default=128, help="Number of instances used for reporting test "
                                                                   "performance")

    # Strategy to Choose CheckPoint
    # Strategy to Choose CheckPoint
    parser.add_argument('--strategy', type=str, choices=['cover_instances', 'best_k', 'consecutive_best_k', 'last_k',
                                                         'single'], default='cover_instances')
    parser.add_argument('--k', type=int, default=5, help="Number of checkpoints to choose.")

    parser.add_argument('--n_mc_sims', default=100, type=int)
    parser.add_argument('--n_process', default=1, type=int, help="When n_process > 1, we start n_process processes to "
                                                                 "evaluate the robustness.")

    args = parser.parse_args()

    # Read Parameters from Model Saving Dir
    opts = load_params(args.model_saving_dir)
    # Do not forget to override test_size and out-of-distribution model
    opts.test_size = args.test_size
    opts.graph_model = args.graph_model
    opts.edge_budget_percentage = args.edge_budget_percentage
    opts.n_mc_sims = args.n_mc_sims

    if opts.method == 'random_removal':
        opts.instantiated_mtd = RandomRemoval
    elif opts.method == 'targeted_removal':
        opts.instantiated_mtd = TargetedRemoval

    opts.largest_possible_ebp = args.largest_possible_ebp

    n_node = int(args.graph_model.split("_")[2])
    largest_n_step = get_edge_budget(n_node, opts.largest_possible_ebp) * 2
    n_step = get_edge_budget(n_node, opts.edge_budget_percentage) * 2

    strategy_config = get_res_name(args.strategy, args.k)
    precomputing_res_name = f'res_{args.graph_model}_{largest_n_step}_{strategy_config}'

    # Pick checkpoints
    precomputing_res_path = os.path.join(args.model_saving_dir, precomputing_res_name)
    assert os.path.exists(precomputing_res_path), f'Precompute {precomputing_res_name} first'

    res = evaluate()
    print(res)
    run_time = time.strftime("%Y%m%dT%H%M%S")
    with open(os.path.join(args.model_saving_dir, f'res_{args.graph_model}_{run_time}_{n_step}_{strategy_config}.txt'),
              'w',
              encoding='utf-8') as output:
        json.dump(vars(args), output, indent=True)
        output.write('\n')
        output.write(res)
