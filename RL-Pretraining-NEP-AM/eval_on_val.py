import os
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
import numpy as np

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
    if not args.use_extra_val_set:
        _, val_seeds, _ = construct_network_seeds(opts.epoch_size, opts.val_size, 0)
    else:
        _, val_seeds, _ = construct_network_seeds(opts.epoch_size + opts.val_size + opts.test_size,
                                                  args.extra_val_set_size, 0)

    val_dataset = EnhancedNetworkDataset(val_seeds, opts)

    if not args.use_extra_val_set:
        assert len(val_dataset) == opts.val_size, "The validation dataset size and validation batch size are unequal~"
        # Initialize DataLoader
        val_dataloader = DataLoader(val_dataset, opts.val_size)
    else:
        val_dataloader = DataLoader(val_dataset, args.extra_val_set_batch_size)


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
        opts.device,
        args.to_double
    ).to(opts.device)

    if args.to_double:
        model.double()

    model.set_decode_type("greedy")
    model.eval()

    # Set baseline=None
    policy = REINFORCE(model, None, opts)

    names, val_imps = [], []
    # Evaluate each checkpoint
    for checkpoint in checkpoints:
        names.append(checkpoint)
        checkpoint = torch.load(os.path.join(model_saving_dir, checkpoint))
        if 'model_state_dict' in checkpoint:
            policy.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            policy.model.load_state_dict(checkpoint)

        val_imps_ckp = []
        # Load the validation dataset
        for graphs, seeds in val_dataloader:
            # numpy array ----> networkx object
            val_graphs = [nx.from_numpy_array(graph.numpy()) for graph in graphs]

            # Construct Graph States
            graphstates = [GraphState(g, opts.edge_budget_percentage, seed=seed) for g, seed in
                           zip(val_graphs, seeds)]

            val_imps_ckp.extend(policy.validate_batch(graphstates).tolist())

        if not args.use_extra_val_set:
            # (n_checkpoints, val_batch_size)
            val_imps.append(val_imps_ckp)
        else:
            val_imps.append([np.array(val_imps_ckp).mean()])

    content = ""
    for name, val_imp in zip(names, val_imps):
        val_imp_str = '\t'.join([f'{_:.4f}' for _ in val_imp])
        content = content + name + "\t" + val_imp_str + "\n"

    return content


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate Models"
    )

    # Validation Set used in the training procedure
    parser.add_argument('--model_saving_dir', type=str, default='./', help="The Directory of Saving Models.")
    parser.add_argument('--no_sim_seed', action='store_true')

    # Extra Validation Set
    parser.add_argument('--test_size', type=int, default=128, help="Number of instances used for reporting test "
                                                                   "performance")
    parser.add_argument('--use_extra_val_set', action='store_true')
    parser.add_argument('--extra_val_set_size', type=int, default=1024, help="Number of extra instances used "
                                                                             "for validation")
    parser.add_argument('--extra_val_set_batch_size', type=int, default=128, help="Batch size for extra validation")

    # Checkpoints chosen by strategy
    parser.add_argument('--strategy', type=str, choices=['cover_instances', 'best_k', 'consecutive_best_k', 'last_k',
                                                         'single'], default=None)
    parser.add_argument('--k', type=int, default=5, help="Number of checkpoints to choose.")

    # Use Float64
    parser.add_argument('--to_double', action='store_true')

    args = parser.parse_args()

    if args.use_extra_val_set:
        res_path = 'res_extra_val'
    else:
        res_path = 'res_val'

    if args.strategy is None:
        res_path = os.path.join(args.model_saving_dir, res_path + '.txt')
    else:
        strategy_config = get_res_name(args.strategy, args.k)
        chk_path = os.path.join(args.model_saving_dir, strategy_config + ".txt")
        res_path = os.path.join(args.model_saving_dir, res_path + '_' + strategy_config + '.txt')

    if os.path.exists(res_path):
        print(f"{res_path} already exists~")
        exit(0)

    # Read Parameters from Model Saving Dir
    opts_ = load_params(args.model_saving_dir)

    if args.no_sim_seed:
        opts_.sim_seed = None

    # Do not forget to override test_size !!!
    opts_.test_size = args.test_size

    if args.strategy is not None:
        assert os.path.exists(chk_path), f'Invoke {args.strategy}.py first'
        checkpoints_ = []
        with open(chk_path, 'r') as in_file:
            checkpoints_ = in_file.readline().strip().split()
    else:
        # Use all checkpoints available
        def get_key(name):
            name = name[0: -3]
            if 'epoch' in name:
                return int(name.split('-')[1])
            elif 'run' in name:
                # Find the epoch id of the checkpoint
                checkpoint = torch.load(os.path.join(args.model_saving_dir, name + ".pt"))
                return checkpoint['epoch_id']
        # Get All the CheckPoints
        checkpoints_ = [file for file in os.listdir(args.model_saving_dir) if file.endswith(".pt")]
        # Sort by the ascending order of epoch_id
        checkpoints_.sort(key=get_key)

    if len(checkpoints_) > 0:
        res = evaluate(args.model_saving_dir, opts_, checkpoints_)
        with open(res_path, 'w', encoding='utf-8') as output:
            output.write(res)
    else:
        print("No Checkpoints Found!")
