import os
import time
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from problems.network_generator import construct_network_seeds
from problems.network_dataset import EnhancedNetworkDataset
from problems.graph_state import GraphState
from nets.attention_model import AttentionModel
from policy.reinforce import REINFORCE
from utils.functions import load_params, get_freer_gpu
import argparse
import networkx as nx
from math import inf
from utils.choose_checkpoint import get_res_name


@torch.no_grad()
def evaluate(model_saving_dir, opts, checkpoints):
    """
    :param model_saving_dir: 
    :param opts:
    :param checkpoints: A List of Names of checkpoints.
    :return:
    """

    # Fetch the Test Data
    # IF the graphs have been stored in disk,
    # they will be loaded directly.
    _, _, test_seeds = construct_network_seeds(opts.epoch_size, opts.val_size, opts.test_size)
    test_dataset = EnhancedNetworkDataset(test_seeds, opts)

    assert len(test_dataset) == opts.test_size, "The test dataset size and test batch size are unequal~"

    # Initialize DataLoader
    test_dataloader = DataLoader(test_dataset, opts.test_size)

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

    # Load the validation dataset
    for graphs, seeds in test_dataloader:
        # numpy array ----> networkx object
        test_graphs = [nx.from_numpy_array(graph.numpy()) for graph in graphs]

        # Construct Graph States
        graphstates = [GraphState(g, opts.edge_budget_percentage, seed=seed) for g, seed in zip(test_graphs, seeds)]

    names, test_imps = [], []
    # Evaluate each checkpoint
    for checkpoint in checkpoints:
        names.append(checkpoint)
        checkpoint = torch.load(os.path.join(model_saving_dir, checkpoint))
        if 'model_state_dict' in checkpoint:
            policy.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            policy.model.load_state_dict(checkpoint)

        # (n_checkpoints, val_batch_size)
        test_imps.append(policy.validate_batch(graphstates))

    # (n_checkpoints, val_batch_size)
    test_imps = np.stack(test_imps, axis=0)
    # (val_batch_size, )
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
    parser.add_argument('--no_sim_seed', action='store_true')
    parser.add_argument('--strategy', type=str, choices=['cover_instances', 'best_k', 'consecutive_best_k', 'last_k',
                                                         'single'], default='cover_instances')
    parser.add_argument('--k', type=int, default=5, help="Number of checkpoints to choose.")
    parser.add_argument('--test_size', type=int, default=128, help="Number of instances used for reporting "
                                                                   "test performance.")

    args = parser.parse_args()

    # Read Parameters from Model Saving Dir
    opts_ = load_params(args.model_saving_dir)

    if args.no_sim_seed:
        opts_.sim_seed = None

    opts_.test_size = args.test_size

    # Read the Chosen CheckPoints
    strategy_config = get_res_name(args.strategy, args.k)
    chk_path = os.path.join(args.model_saving_dir, strategy_config + ".txt")
    assert os.path.exists(chk_path), f'Invoke {args.strategy}.py first'
    checkpoints_ = []
    with open(chk_path, 'r') as in_file:
        checkpoints_ = in_file.readline().strip().split()

    if len(checkpoints_) > 0:
        res = evaluate(args.model_saving_dir, opts_, checkpoints_)
        print(res)
        run_time = time.strftime("%Y%m%dT%H%M%S")
        with open(os.path.join(args.model_saving_dir,
                  f'res_{opts_.graph_model}_{run_time}_{strategy_config}.txt'),
                  'w',
                  encoding='utf-8') as output:
            output.write(res)
    else:
        print("No Checkpoints Found!")
