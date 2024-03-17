import argparse
import os
import pickle
from math import ceil

import networkx as nx
import torch
from mpire import WorkerPool
from torch.utils.data.dataloader import DataLoader

from nets.attention_model import AttentionModel
from policy.reinforce import REINFORCE
from problems.graph_state import GraphState
from problems.network_dataset import EnhancedNetworkDataset
from problems.network_enhancement_env import NetworkEnhancementEnv
from problems.network_generator import construct_network_seeds
from problems.random_removal import RandomRemoval
from problems.targeted_removal import TargetedRemoval
from utils.choose_checkpoint import get_res_name
from utils.functions import load_params, get_freer_gpu


def get_edge_budget(num_node, edge_budget_percentage):
    edge_budget = ceil((num_node * (num_node - 1)) / 2.0 * edge_budget_percentage / 100.0)
    return edge_budget


def get_modifications_parallel(seeds, env, checkpoints):
    # {checkpoint : {seed: [ u, v ] }}
    modifications = {}

    print(f'Computing Modifications~')
    with WorkerPool(n_jobs=args.n_process, start_method='spawn') as pool:
        selected_all_steps = pool.map(_compute_modifications_parallel,
                                      zip(checkpoints, [env] * len(checkpoints),
                                          [opts] * len(checkpoints),
                                          list(range(args.n_process))))

    for idx, checkpoint in enumerate(checkpoints):
        modifications[checkpoint] = dict(zip(seeds, selected_all_steps[idx].tolist()))

    return modifications


def get_modifications(seeds, env, checkpoints):
    # {checkpoint : {seed: [ u, v ] }}
    modifications = {}
    for checkpoint in checkpoints:
        modifications[checkpoint] = {}
        # (batch_size, step_num)
        # env wouldn't be changed ~
        print(f'Computing Modifications for :{checkpoint}')
        selected_all_steps = _compute_modifications(checkpoint, env).tolist()
        modifications[checkpoint] = dict(zip(seeds, selected_all_steps))
        print(f'Finish Computing Modifications for :{checkpoint}')

    return modifications


@torch.no_grad()
def _compute_modifications(checkpoint, env):
    """
    :param checkpoint: Name of a checkpoint
    :param env:
    :return:
    """

    # Set the device
    if opts.use_cuda:
        opts.device = torch.device(get_freer_gpu())
    else:
        opts.device = "cpu"

    # Instantiate the Model
    model = AttentionModel(
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
    model.set_decode_type("greedy")
    model.eval()

    # Set baseline=None
    policy = REINFORCE(model, None, opts)

    # Rollout the checkpoint
    checkpoint = torch.load(os.path.join(args.model_saving_dir, checkpoint))
    if 'model_state_dict' in checkpoint:
        policy.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.model.load_state_dict(checkpoint)

    # (batch_size, step_num)
    _, selected_all_steps, _, _ = policy.model(env)

    return selected_all_steps


@torch.no_grad()
def _compute_modifications_parallel(checkpoint, env, opts, process_id):
    """
    :param checkpoint: Name of a checkpoint
    :param env:
    :param opts:
    :param process_id:
    :return:
    """

    # Set the device
    if opts.use_cuda:
        opts.device = torch.device(get_freer_gpu(process_id))
    else:
        opts.device = "cpu"

    # Instantiate the Model
    model = AttentionModel(
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
    model.set_decode_type("greedy")
    model.eval()

    # Set baseline=None
    policy = REINFORCE(model, None, opts)

    # Rollout the checkpoint
    checkpoint = torch.load(os.path.join(opts.model_saving_dir, checkpoint))
    if 'model_state_dict' in checkpoint:
        policy.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.model.load_state_dict(checkpoint)

    # (batch_size, step_num)
    _, selected_all_steps, _, _ = policy.model(env)

    return selected_all_steps


@torch.no_grad()
def precompute(checkpoints):
    """
    :param checkpoints: Name List of Checkpoints.
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

    # Compute the modifications
    # {checkpoint : {seed: [ u, v ] }}
    if args.n_process == 1:
        return get_modifications(seeds, env, checkpoints)
    else:
        return get_modifications_parallel(seeds, env, checkpoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Precompute Modifications"
    )

    parser.add_argument('--model_saving_dir', type=str, default='./', help="The Directory of Saving Models.")
    parser.add_argument('--graph_model', type=str, default='BA_n_50_m_2', help="The graph model (out-of-distribution) "
                                                                               "to evaluate")
    parser.add_argument('--largest_possible_ebp', type=float, default='1.0')

    parser.add_argument('--test_size', type=int, default=5, help="Number of instances used for reporting test "
                                                                 "performance")

    # Strategy to Choose CheckPoint
    parser.add_argument('--strategy', type=str, choices=['cover_instances', 'best_k', 'consecutive_best_k', 'last_k',
                                                         'single'], default='cover_instances')
    parser.add_argument('--k', type=int, default=5, help="Number of checkpoints to choose.")

    parser.add_argument('--n_process', type=int, default=2, help="Number of concurrent process. When n_process=1, we "
                                                                 "do not use the multiprocess version.")

    args = parser.parse_args()

    # Read Parameters from Model Saving Dir
    opts = load_params(args.model_saving_dir)
    # Do not forget to override test_size and out-of-distribution model
    opts.test_size = args.test_size
    opts.graph_model = args.graph_model
    opts.edge_budget_percentage = args.largest_possible_ebp

    if opts.method == 'random_removal':
        opts.instantiated_mtd = RandomRemoval
    elif opts.method == 'targeted_removal':
        opts.instantiated_mtd = TargetedRemoval

    opts.largest_possible_ebp = args.largest_possible_ebp
    # Used in Multiprocess Version
    opts.model_saving_dir = args.model_saving_dir

    n_node = int(args.graph_model.split("_")[2])
    n_step = get_edge_budget(n_node, opts.largest_possible_ebp) * 2

    res_file_name = f'res_{args.graph_model}_{n_step}_{get_res_name(args.strategy, args.k)}'
    res_file_path = os.path.join(args.model_saving_dir, res_file_name)
    if not os.path.exists(res_file_path):

        # Read the Chosen CheckPoints
        chk_path = os.path.join(args.model_saving_dir, get_res_name(args.strategy, args.k) + ".txt")
        assert os.path.exists(chk_path), f'Invoke {args.strategy}.py first'
        checkpoints_ = None
        with open(chk_path, 'r') as in_file:
            checkpoints_ = in_file.readline().strip().split()

        modifications = precompute(checkpoints_)

        with open(res_file_path, 'wb') as output:
            pickle.dump(modifications, output)
        print(f'Finish Computing Modifications for :{res_file_name}')
    else:
        print(f'{res_file_path} already exists~')
