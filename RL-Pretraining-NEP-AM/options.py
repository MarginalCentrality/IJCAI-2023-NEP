import argparse
import torch
import time
import os


def get_options(args=None, resume=False):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Network Enhancement Problem with Reinforcement Learning"
    )

    # Data
    # [model_name]_n_[graph_size] + other params
    # model_name : BA, GNM
    parser.add_argument('--graph_model', type=str, default='BA_n_20_m_2', help="The graph model to use")
    parser.add_argument('--store_graphs', action='store_true')
    parser.add_argument('--graph_storage_root', type=str, default='data')

    parser.add_argument('--batch_size', type=int, default=2 ** 7, help="Number of instances per batch during training")
    parser.add_argument('--epoch_size', type=int, default=2 ** 8, help="Number of instances per epoch during training")

    parser.add_argument('--val_batch_size', type=int, default=2 ** 4, help="Batch size to use during (baseline) "
                                                                           "validation ")
    parser.add_argument('--val_size', type=int, default=2 ** 4, help="Number of instances used for reporting "
                                                                     "validation performance")

    parser.add_argument('--test_size', type=int, default=32, help="Number of instances used for reporting test "
                                                                  "performance")

    parser.add_argument('--delta_seed', type=int, default=10 ** 9, help="When a disconnected graph is generated by "
                                                                        "seed, we will try seed + delta_seed, "
                                                                        "seed + 2 * delta_seed in turn until a "
                                                                        "connected graph is generated.")

    # Model
    parser.add_argument('--node_dim', type=int, default=2, help="Dimension of shallow embedding")

    parser.add_argument('--embedding_dim', type=int, default=32, help="Dimension of encoder's input/output embedding")
    parser.add_argument('--feed_forward_hidden', type=int, default=512,
                        help="Dimension of hidden vector in feed forward layer")
    parser.add_argument('--n_encode_layers', type=int, default=3, help='Number of layers in the encoder network')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads in Multi-Head Attention layer')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help="Clip the logits to within +- this value using tanh. "
                             "Set to 0 to not perform any clipping.")

    parser.add_argument('--aggr_func', type=str, default='sum', help="Aggregation Method for network representation: "
                                                                     "sum or mean")

    parser.add_argument('--pos_enc', choices=['fix', 'approx', 'dynamic'], default='dynamic')

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="Learning rate decay per epoch")
    parser.add_argument('--n_epochs', type=int, default=5000, help="The number of epochs to train")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed to use")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum L2 norm for gradient clipping")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8, help='Exponential moving average baseline decay ('
                                                                    'default 0.8)')
    parser.add_argument('--baseline', default=None, help='Baseline to use: rollout, or exponential. Defaults to no'
                                                         'baseline.')
    parser.add_argument('--bl_reject_threshold', type=float, default=0.05, help="Significance in the t-test for "
                                                                                "updating rollout baseline")

    parser.add_argument('--bl_warmup_epochs', type=int, default=0, help="warmup epochs for rollout (exponential "
                                                                        "used for warmup phase)")

    parser.add_argument('--method', type=str, default='targeted_removal', help="Attack method: random_removal, or "
                                                                               "targeted_removal")

    parser.add_argument('--no_sim_seed', action='store_true')
    parser.add_argument('--sim_seed', type=int, default=None, help='Seed to use for random attack.')
    parser.add_argument('--reuse_hash', action='store_true')

    parser.add_argument('--edge_budget_percentage', type=float, default=20, help="The unit is percentage.")
    parser.add_argument('--n_mc_sims', type=int, default=100, help="Number of MC simulations used "
                                                                   "in random_removal")
    parser.add_argument('--scale_factor', type=float, default=100, help="Scale the loss function by scale_factor")

    # Misc
    # parser.add_argument('--log_step', type=int, default=10, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--checkpoint_batches', type=int, default=10, help='Evaluate the model every n batches')
    parser.add_argument('--checkpoint_epochs', type=int, default=100, help='Save a general checkpoint every n epochs')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging Tensorboard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    # These parameters are useless, only to avoid error.
    # When loading params from args.json, the args contains 'save_dir' and 'use_cuda'
    if resume:
        parser.add_argument('--save_dir', type=str, help='Directory to read output models')
        parser.add_argument('--use_cuda', action='store_true')

    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda

    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        opts.graph_model,
        opts.run_name
    )

    assert opts.epoch_size % opts.batch_size == 0, 'Epoch size must be integer multiple of batch size!'
    assert opts.val_size % opts.val_batch_size == 0, 'Validation size must be integer multiple of validation batch ' \
                                                     'size! '
    assert opts.bl_warmup_epochs == 0 or opts.baseline == 'rollout', 'We can only warm up rollout baseline.'

    return opts


if __name__ == '__main__':
    opts = get_options(['--graph_model', 'Random'])
    print(opts.graph_model)