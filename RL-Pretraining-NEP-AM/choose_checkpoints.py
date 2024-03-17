import argparse
import os
from copy import copy

import numpy as np
from utils.choose_checkpoint import read_validation_res, write_chosen_chk_points_to_disk, get_res_name


def choose_chk_to_cover_instances(checkpoints, val_imps):
    """
    Choose k checkpoints to cover instances as much as possible
    Args:
        checkpoints: Name list of checkpoints
        val_imps: Validation results of checkpoints. An array of size (n_checkpoints, val_size).
    Returns:
        Name list of chosen checkpoints.
    """

    checkpoints = copy(checkpoints)

    # (val_size, )
    baseline = np.zeros(val_imps.shape[1])
    chosen = []
    while len(chosen) < args.k:
        # (n_remain_checkpoints, val_size)
        relative_inc = val_imps - baseline
        # (n_remain_checkpoints, val_size)
        relative_inc[relative_inc < 0] = 0
        idx = np.argmax(np.sum(relative_inc, axis=1))

        # update baseline
        # (2, val_size) -> (val_size, )
        baseline = np.stack([val_imps[idx], baseline], axis=0).max(axis=0)

        chosen.append(checkpoints[idx])

        # update checkpoints and val_imps
        del checkpoints[idx]
        val_imps = np.delete(val_imps, idx, axis=0)

    return chosen


def choose_best_k_checkpoints():
    """
    Choose k checkpoints with the highest average validation improvements.
    Returns:
        Name list of chosen checkpoints.
    """

    # (n_checkpoints, val_size) -> (n_checkpoints, )
    avg_val_imps = val_imps.mean(axis=1)
    # In ascending order, (n_checkpoints, )
    sorted_index_array = np.argsort(avg_val_imps)

    chosen = []
    for idx in sorted_index_array[-args.k:]:
        chosen.append(checkpoints[idx])

    return chosen


def choose_consecutive_best_k_checkpoints():
    """
    Choose k consecutive checkpoints with the highest average validation improvements.
    Returns:
        Name list of chosen checkpoints.
    """
    # (n_checkpoints, val_size) -> (n_checkpoints, )
    avg_val_imps = val_imps.mean(axis=1)

    consecutive_val_imps = []
    for i in range(0, avg_val_imps.size - args.k + 1):
        consecutive_val_imps.append(np.sum(avg_val_imps[i: i + args.k]))

    idx = np.argmax(np.array(consecutive_val_imps))

    chosen = [checkpoints[_] for _ in range(idx, idx + args.k)]

    return chosen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Choose Checkpoints Greedily~"
    )

    parser.add_argument('--model_saving_dir', type=str, default='./', help="Directory of Saving Models.")
    parser.add_argument('--strategy', type=str, choices=['cover_instances', 'best_k', 'consecutive_best_k', 'last_k',
                                                         'single'], default='cover_instances')
    parser.add_argument('--k', type=int, default=5, help="Number of checkpoints to choose.")

    args = parser.parse_args()

    res_name = get_res_name(args.strategy, args.k)
    res_path = os.path.join(args.model_saving_dir, res_name+'.txt')
    if os.path.exists(res_path):
        print(f'{res_path} already exists~')
        exit(0)

    # Read file
    # checkpoints : Name list of checkpoints
    # val_imps : Validation results of checkpoints. An array of size (n_checkpoints, val_size)
    checkpoints, val_imps = read_validation_res(args.model_saving_dir, 'res_val.txt')

    if args.strategy == 'cover_instances':
        res = choose_chk_to_cover_instances(checkpoints, val_imps)
    elif args.strategy == 'best_k':
        res = choose_best_k_checkpoints()
    elif args.strategy == 'consecutive_best_k':
        res = choose_consecutive_best_k_checkpoints()
    elif args.strategy == 'last_k':
        res = checkpoints[-args.k:]
    elif args.strategy == 'single':
        for chk in checkpoints:
            if 'run' in chk:
                res = [chk]
                break

    # Write res into disk
    write_chosen_chk_points_to_disk(args.model_saving_dir, res_name + '.txt', res)
