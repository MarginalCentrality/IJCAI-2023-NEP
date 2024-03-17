import os
import json
import torch
from torch.utils.data.dataloader import DataLoader
from tensorboard_logger import Logger as TbLogger
import pprint as pp
from problems.network_generator import construct_network_seeds
from problems.network_dataset import NetworkDataset, EnhancedNetworkDataset
from nets.attention_model import AttentionModel
from reinforce_baselines.baselines import NoBaseLine, ExponentialBaseline, RolloutBaseline, \
    WarmupBaseline
from policy.reinforce import REINFORCE
from utils.functions import load_params
from copy import deepcopy
import argparse


def resume(opts, resume_dir, resume_epoch_id):
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Save arguments
    os.makedirs(opts.save_dir)
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

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

    # Recover Data
    # IF the graphs have been stored in disk,
    # they will be loaded directly.
    training_seeds, val_seeds, test_seeds = construct_network_seeds(opts.epoch_size, opts.val_size, opts.test_size)
    training_dataset = EnhancedNetworkDataset(training_seeds, opts)
    val_dataset = EnhancedNetworkDataset(val_seeds, opts)

    # Initialize DataLoader
    train_dataloader = DataLoader(training_dataset, opts.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, opts.val_batch_size)

    # Recover the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Read the checkpoint
    checkpoint_path = os.path.join(resume_dir, 'epoch'+'-'+str(resume_epoch_id)+'.pt')
    assert os.path.exists(checkpoint_path), f"{checkpoint_path} does not exist."
    checkpoint = torch.load(checkpoint_path)

    # Recover Epoch ID
    epoch_to_conti = checkpoint['epoch_id'] + 1

    # Recover the Model
    # # Initialize a Model
    model = AttentionModel(
        opts.node_dim,
        opts.embedding_dim,
        opts.feed_forward_hidden,
        opts.n_encode_layers,
        opts.n_heads,
        opts.tanh_clipping,
        opts.device
    ).to(opts.device)
    # # Recover the parameters.
    model.load_state_dict(checkpoint['model_state_dict'])

    # # TODO: Make the Model to be DataParallel

    if opts.bl_warmup_epochs > 0:
        baseline_state_dict = checkpoint['baseline_state_dict']['baseline']
    else:
        baseline_state_dict = checkpoint['baseline_state_dict']

    # Initialize baseline
    if opts.baseline is None:
        baseline = NoBaseLine()
    elif opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
        baseline.b_val = baseline_state_dict['b_val']
    elif opts.baseline == 'rollout':
        baseline_model = deepcopy(model)
        baseline_model.load_state_dict(baseline_state_dict['model_state_dict'])
        init_seed = baseline_state_dict['seed_to_use']
        baseline = RolloutBaseline(baseline_model, init_seed, opts)
    else:
        raise NotImplementedError(f"{opts.baseline} has not been implemented")

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, opts.exp_beta)
        baseline.n_callbacks = checkpoint['baseline_state_dict']['n_callbacks']
        baseline.warmup_baseline.b_val = checkpoint['baseline_state_dict']['b_val']

    # Recover the policy
    policy = REINFORCE(model, baseline, opts, tb_logger)
    # # Recover the optimizer
    policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # # Recover the LR scheduler
    policy.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # # Recover the best_val
    policy.best_val = checkpoint['best_val']

    # Recover the random state
    torch.set_rng_state(checkpoint['rng_state'])
    if opts.use_cuda:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    # Resume training process
    for epoch_id in range(epoch_to_conti, opts.n_epochs):
        policy.train_epoch(epoch_id,
                           train_dataloader,
                           val_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Resume the training procedure."
    )

    parser.add_argument('--resume_dir', type=str, default='./', help="The Directory of An Instance")
    parser.add_argument('--resume_epoch_id', type=int, default=0, help="Continue at resume_epoch_id + 1")

    args = parser.parse_args()

    # Read Parameters from resume_dir
    opts = load_params(args.resume_dir)

    if args.resume_epoch_id + 1 < opts.n_epochs:
        resume(opts, args.resume_dir, args.resume_epoch_id)
    else:
        print("No need to resume running.")
