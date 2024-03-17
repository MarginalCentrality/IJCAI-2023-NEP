import os
import numpy as np


def read_validation_res(res_dir, res_name):
    """
    Args:
        res_dir:  Directory to store the 'res_val.txt'
        res_name: 'res_val.txt'

    Returns:
        Name list of checkpoints
        Validation results of checkpoints. An array of size (n_checkpoints, val_size)
    """
    res_path = os.path.join(res_dir, res_name)
    with open(res_path, 'r') as input:
        checkpoints, vals = [], []
        for line in input:
            contents = line.strip().split()
            checkpoints.append(contents[0])
            vals.append(contents[1:])

    return checkpoints, np.asarray(vals, dtype=float)


def write_chosen_chk_points_to_disk(res_dir, strategy_name, chosen_checkpoints):
    """
    Args:
        res_dir: Directory
        strategy_name:
        chosen_checkpoints: Name list of chosen checkpoints.

    Returns:

    """
    with open(os.path.join(res_dir, strategy_name), 'w') as output:
        content = '\t'.join(chosen_checkpoints)
        output.write(content)


def get_res_name(strategy, k):
    """
    Args:
        strategy: name of the strategy
        k: number of chosen checkpoints

    Returns:
        name of the res
    """
    res_name = None
    if strategy == 'cover_instances':
        res_name = strategy + f'_{k}'
    elif strategy == 'best_k':
        res_name = f'best_{k}'
    elif strategy == 'consecutive_best_k':
        res_name = f'consecutive_best_{k}'
    elif strategy == 'last_k':
        res_name = f'last_{k}'
    elif strategy == 'single':
        res_name = strategy

    return res_name




