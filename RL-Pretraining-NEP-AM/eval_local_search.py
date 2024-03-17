import os
import time
import json
import torch
from torch.utils.data.dataloader import DataLoader

from problems.graph_state import GraphState
from problems.network_generator import construct_network_seeds
from problems.network_dataset import NetworkDataset
from utils.functions import load_params, get_freer_gpu
import argparse
from argparse import Namespace
import networkx as nx
import heapq
from math import sqrt
from math import inf
import numpy as np
from tabulate import tabulate
from problems.network_enhancement_env import NetworkEnhancementEnv
from heuristics.local_search import LocalSearch
from heuristics.beam_search import BeamSearch, BeamSearchModel
from heuristics.random_search import RandomSearch
from nets.attention_model import AttentionModel
from problems.targeted_removal import TargetedRemoval
from problems.random_removal import RandomRemoval
from mpire import WorkerPool
import warnings
from utils.analyze_ls_results import get_avg_imp, get_max_imp, get_n_diff_optimal_sols


warnings.filterwarnings("ignore")


def eval_local_search(model_saving_dir, opts, checkpoints):
    """
    :param model_saving_dir:
    :param opts:
    :param checkpoints: List. Each item is a name of checkpoint.
    :return:
    """

    # Fetch the Test Data
    # IF the graphs have been stored in disk,
    # they will be loaded directly.
    _, _, test_seeds = construct_network_seeds(opts.epoch_size, opts.val_size, opts.test_size)
    test_dataset = NetworkDataset(test_seeds, opts)

    assert len(test_dataset) == opts.test_size, "The test dataset size and test batch size are unequal~"

    # Initialize DataLoader and Process 1 Network Each Time.
    test_dataloader = DataLoader(test_dataset, 1)

    # Set the device
    if opts.use_cuda:
        opts.device = torch.device(get_freer_gpu())
    else:
        opts.device = "cpu"

    # Instantiate the beam search model
    beam_search_model = BeamSearchModel(
                        opts.node_dim,
                        opts.embedding_dim,
                        opts.pos_enc,
                        opts.feed_forward_hidden,
                        opts.n_encode_layers,
                        opts.n_heads,
                        opts.tanh_clipping,
                        opts.aggr_func,
                        opts.device
                        ).to(opts.device)

    beam_search_model.set_decode_type("greedy")
    beam_search_model.eval()

    random_search_model = None

    if opts.init_sol_mtd == "random_search":
        # Instantiate the random search model
        random_search_model = AttentionModel(
                                opts.node_dim,
                                opts.embedding_dim,
                                opts.pos_enc,
                                opts.feed_forward_hidden,
                                opts.n_encode_layers,
                                opts.n_heads,
                                opts.tanh_clipping,
                                opts.aggr_func,
                                opts.device
                                ).to(opts.device)
        random_search = RandomSearch(random_search_model, seed=opts.random_search_seed)
    else:
        # Initialize the Beam Search.
        beam = BeamSearch(beam_search_model,
                          opts.beam_width_init_sol)

    # Instantiate the Local Search
    localsearch = LocalSearch(beam_search_model,
                              opts.swap_size,
                              opts.swap_step_size,
                              opts.beam_width,
                              opts.nbor_size,
                              opts.method,
                              opts.n_mc_sims,
                              opts.sim_seed,
                              opts.reuse_hash)

    # Record Avg Improvements
    avg_imps = []

    # Instantiate the Envs
    envs = []
    init_robustness = []
    for graphs in test_dataloader:
        # numpy array ----> networkx object
        test_graphs = [nx.from_numpy_array(graph.numpy()) for graph in graphs]

        # Construct Graph States
        graphstates = [GraphState(g,
                                  opts.edge_budget_percentage)
                       for g in test_graphs]

        # Get the environment
        # Each env contains only one graph.
        env = NetworkEnhancementEnv(graphstates)

        envs.append(env)
        init_robustness.extend(env.calculate_robustness(opts.method, opts.n_mc_sims, opts.sim_seed, opts.reuse_hash))

    init_robustness = np.array(init_robustness)

    avg_env_maximal_imps = []
    avg_env_avg_imps = []
    avg_env_n_diff_opt_sols = []
    for checkpoint in checkpoints:
        print(f'Begin Evaluating Local Search on Checkpoint: {checkpoint}')
        checkpoint = torch.load(os.path.join(model_saving_dir, checkpoint))
        if 'model_state_dict' in checkpoint:
            if random_search_model is not None:
                random_search_model.load_state_dict(checkpoint['model_state_dict'])
            beam_search_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            if random_search_model is not None:
                random_search_model.load_state_dict(checkpoint)
            beam_search_model.load_state_dict(checkpoint)

        maximal_imps = []  # The maximal improvement for each env
        avg_imps = []  # The avg improvements of all solutions for each env
        n_diff_opt_sols = []  # The number of different optimal solutions for each env

        # Load the test dataset
        for idx, env in enumerate(envs):
            # Get the initial solutions
            # (log prob of the init_sol, node list)

            if opts.init_sol_mtd == "random_search":
                init_sols = random_search.random_search(env, opts.n_init_sol)
            else:
                init_sols = beam.beam_search(env)
                heapq.heapify(init_sols)
                init_sols = [item[1] for item in heapq.nlargest(opts.n_init_sol, init_sols)]

            # Improve the initial solutions
            # Set start_method='spawn' to avoid "Cannot re-initialize CUDA in forked subprocess."
            with WorkerPool(n_jobs=opts.n_child_procs, start_method='spawn') as pool:
                # An imp_sol contains (robustness, node list)
                if not opts.advanced_local_search:
                    imp_sols = pool.map(localsearch.local_search, zip(init_sols, [env] * len(init_sols)))
                else:
                    imp_sols = pool.map(localsearch.advanced_local_search, zip(init_sols, [env] * len(init_sols),
                                                                               [opts.swap_size_list] * len(init_sols)))

            # Compute statistics
            maximal_imps.append(get_max_imp(init_robustness[idx], imp_sols))
            avg_imps.append(get_avg_imp(init_robustness[idx], imp_sols))
            n_diff_opt_sols.append(get_n_diff_optimal_sols(imp_sols))

        avg_env_maximal_imps.append(np.mean(np.array(maximal_imps)))
        avg_env_avg_imps.append(np.mean(np.array(avg_imps)))
        avg_env_n_diff_opt_sols.append(np.mean(np.array(n_diff_opt_sols)))

    # Generate the Table
    table_header = ['checkpoint', 'avg_env_maximal_imps', 'avg_env_avg_imps', 'avg_env_n_diff_opt_sols']
    res_table = tabulate(zip(checkpoints, avg_env_maximal_imps, avg_env_avg_imps, avg_env_n_diff_opt_sols),
                         headers=table_header,
                         tablefmt='fancy_grid',
                         floatfmt=".3f",
                         showindex=True)

    return res_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Local Search"
    )

    parser.add_argument('--model_saving_dir', type=str, default='./', help="The Directory of Saving Models.")
    parser.add_argument('--test_size', type=int, default=1, help="Number of instances to Test Local Search.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Checkpoint to use (end with .pt)."
                                                                     "Default to use all checkpoints"
                                                                     "under the model_saving_dir.")

    parser.add_argument('--init_sol_mtd', type=str, choices=['random_search', 'beam_search'])
    parser.add_argument('--random_search_seed', type=int, default=None, help="Seed to control the generation of "
                                                                             "initial solutions.")

    parser.add_argument('--beam_width_init_sol', type=int, default=1, help="Beam Width for Searching the Initial"
                                                                           "Solution.")
    parser.add_argument('--n_init_sol', type=int, default=8, help="The number of the initial solutions.")
    parser.add_argument('--n_child_procs', type=int, default=1, help="The number of child processes.")

    parser.add_argument('--swap_size', type=int, default=1, help="n_swap-OPT. "
                                                                 "In each swap, n_swap nodes would be swapped out.")

    parser.add_argument('--swap_step_size', type=int, default=1, help="Interval between the swap starting points.")

    parser.add_argument('--beam_width', type=int, default=3, help="Beam Search width.")
    parser.add_argument('--nbor_size', type=int, default=10, help="Size of the neighborhood.")

    parser.add_argument('--no_sim_seed', action='store_true')
    parser.add_argument('--reuse_hash', action='store_true')

    parser.add_argument('--advanced_local_search', action='store_true')
    parser.add_argument('--swap_size_list', nargs='*')

    args = parser.parse_args()

    # IF we use advanced_local_search, then swap_size_list cannot be None
    assert not args.advanced_local_search or args.swap_size_list is not None
    if args.swap_size_list is not None:
        args.swap_size_list = [int(_) for _ in args.swap_size_list]

    # Get the res name
    checkpoint_name = 'ALL' if args.checkpoint is None else args.checkpoint[0:-3]
    run_time = time.strftime("%Y%m%dT%H%M%S")

    if not args.advanced_local_search:
        res_name = f'{checkpoint_name}-{run_time}-localsearch.txt'
    else:
        res_name = f'{checkpoint_name}-{run_time}-adv-localsearch.txt'

    # Read Parameters from Model Saving Dir
    opts_ = load_params(args.model_saving_dir)

    # Update Params
    opts_dict = vars(opts_)
    args_dict = vars(args)
    opts_dict.update(args_dict)
    opts_ = Namespace(**opts_dict)

    if opts_.method == 'random_removal':
        opts_.method = RandomRemoval
    elif opts_.method == 'targeted_removal':
        opts_.method = TargetedRemoval

    if opts_.no_sim_seed:
        opts_.sim_seed = None

    if opts_.checkpoint is None:
        def get_key(name):
            name = name[0: -3]
            if 'epoch' in name:
                return int(name.split('-')[1])
            elif 'run' in name:
                return inf


        # Get All the CheckPoints
        checkpoints_ = [file for file in os.listdir(args.model_saving_dir) if file.endswith(".pt")]
        checkpoints_.sort(key=get_key)
    else:
        checkpoints_ = [opts_.checkpoint]

    if len(checkpoints_) > 0:
        res_table = eval_local_search(args.model_saving_dir, opts_, checkpoints_)
        with open(os.path.join(args.model_saving_dir, res_name), 'w', encoding='utf-8') as output:
            json.dump(args_dict, output, indent=True)
            output.write('\n' + res_table)
    else:
        print("No Checkpoints Found!")
