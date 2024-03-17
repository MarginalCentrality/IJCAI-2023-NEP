import os
import json
import torch
from torch.utils.data.dataloader import DataLoader
from tensorboard_logger import Logger as TbLogger
import pprint as pp
from options import get_options
from problems.network_generator import construct_network_seeds
from problems.network_dataset import NetworkDataset, EnhancedNetworkDataset
from nets.attention_model import AttentionModel
from reinforce_baselines.baselines import NoBaseLine, ExponentialBaseline, RolloutBaseline, \
    WarmupBaseline
from policy.reinforce import REINFORCE
from utils.functions import clock, get_freer_gpu


def run(opts):
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(
            os.path.join(
                opts.log_dir,
                opts.graph_model,
                opts.run_name
            )
        )

    # Save arguments
    os.makedirs(opts.save_dir)
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Get Data.
    # IF the graphs have been stored in disk,
    # they will be loaded directly.
    training_seeds, val_seeds, test_seeds = construct_network_seeds(opts.epoch_size, opts.val_size, opts.test_size)
    training_dataset = EnhancedNetworkDataset(training_seeds, opts)
    val_dataset = EnhancedNetworkDataset(val_seeds, opts)

    # Initialize DataLoader
    train_dataloader = DataLoader(training_dataset, opts.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, opts.val_batch_size)

    # Set the device
    if opts.use_cuda:
        opts.device = torch.device(get_freer_gpu())
    else:
        opts.device = torch.device("cpu")

    # Initialize Model
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

    # TODO: Make the Model be DataParallel
    # TODO: Load params from Disk or Not.

    # Initialize baseline
    if opts.baseline is None:
        baseline = NoBaseLine()
    elif opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        init_seed = opts.epoch_size + opts.val_size + opts.test_size
        baseline = RolloutBaseline(model, init_seed, opts)
    else:
        raise NotImplementedError(f"{opts.baseline} has not been implemented")

    # From here to continue
    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, opts.exp_beta)

    # Initialize the policy
    policy = REINFORCE(model, baseline, opts, tb_logger)

    for epoch_id in range(opts.n_epochs):
        policy.train_epoch(epoch_id,
                           train_dataloader,
                           val_dataloader)


if __name__ == '__main__':
    opts_ = get_options(['--graph_model', 'BA_n_20_m_2',
                         '--store_graphs',
                         '--graph_storage_root', '../network-enhancement',
                         '--n_epochs', '10',
                         '--checkpoint_epochs', '2',
                         '--baseline', 'rollout',
                         '--bl_warmup_epochs', '0',
                         '--max_grad_norm', 'inf',
                         '--edge_budget_percentage', '2.5',
                         '--lr_model', '1e-4',
                         '--no_progress_bar',
                         '--aggr_func', 'mean',
                         '--method', 'targeted_removal',
                         '--sim_seed', '42',
                         '--n_mc_sims', '10',
                         '--pos_enc', 'fix'
                         ])
    # opts_ = get_options()
    run(opts_)
