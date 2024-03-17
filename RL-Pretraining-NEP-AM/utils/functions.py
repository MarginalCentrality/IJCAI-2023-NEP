import json
import os

import torch.cuda

from options import get_options
import time
import numpy as np
import csv


def list2dict(lst):
    """
    Convert a list to dict.
    Convert [e[0], e[1], ..., e[2i], e[2i+1], ...] to {..., e[2i]: e[2i+1], ...}.
    :param lst:
    :return:
    """
    assert len(lst) % 2 == 0, 'List should contain an even number of elements'
    it = iter(lst)
    return dict(zip(it, it))


def load_params(resume_dir):
    """
    :param resume_dir:
    :return:
    """
    args_path = os.path.join(resume_dir, 'args.json')
    assert os.path.exists(args_path), f"The path {args_path} does not exists."
    with open(args_path, 'r') as f:
        args = json.load(f)

    opts = []
    for k, v in args.items():
        opts.append('--' + k)
        # None for argument baseline
        if v is False or v is None:
            opts.pop()
        elif v is not True:
            opts.append(str(v))

    return get_options(opts, resume=True)


def clock(tag=None):
    def inner(func):
        def clocked(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            name = func.__name__
            if tag is not None:
                name = str(tag) + ':' + name
            print(f'[{elapsed:.8f}s] -> {name}')
            return result

        return clocked

    return inner


def get_freer_gpu(process_id=None):
    """
    Args:
        process_id: ID number of a process.

    Returns:
    """
    gpu_id = None
    suffix = ""
    if process_id is not None:
        suffix = str(process_id)

    tmp = f'_tmp_{suffix}'
    if torch.cuda.is_available():
        os.system(f'nvidia-smi -q -d Memory | grep -A4 GPU| grep Free > {tmp}')
        with open(tmp, 'r') as infile:
            mem_available = [int(line.split()[2]) for line in infile.readlines()]
        os.remove(tmp)

        max_mem_available = max(mem_available)
        gpu_id = mem_available.index(max_mem_available)

    return "cuda:" + str(gpu_id)


def parse_res_table(res_table_path, col_id_epoch_id=3, col_id_val_avg_imp=4, col_id_test_avg_imp=7, splitter='│'):
    # Skip the table border
    skip = ['╒═══', '╞═══', '├───', '╘═══']

    # Filter the table border from the table.
    contents = []
    with open(res_table_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for idx, row in enumerate(csv_reader):
            content = row[0]
            to_skip = map(lambda sub_str: sub_str in content, skip)
            if any(to_skip):
                continue

            contents.append(content)

    # Make Sure the column indexes in parameters are right.
    header = contents[0].split(splitter)
    assert 'epoch_id' in header[col_id_epoch_id] \
           and 'Val Avg Imp' in header[col_id_val_avg_imp] \
           and 'Test Avg Imp' in header[col_id_test_avg_imp], 'Error in Column indexes'

    # Skip the header
    contents = contents[1:]

    # Skip the last 2 rows
    contents = contents[:-2]

    res_len = len(contents)
    epoch_ids = [None] * res_len
    val_avg_imps = [None] * res_len
    test_avg_imps = [None] * res_len

    for idx, content in enumerate(contents):
        content = content.split(splitter)
        epoch_ids[idx], val_avg_imps[idx], test_avg_imps[idx] = int(content[col_id_epoch_id].strip()), \
                                                                float(content[col_id_val_avg_imp].strip()), \
                                                                float(content[col_id_test_avg_imp].strip())

    epoch_ids, val_avg_imps, test_avg_imps = np.array(epoch_ids, dtype=int), \
                                             np.array(val_avg_imps, dtype=float), \
                                             np.array(test_avg_imps, dtype=float)

    # Sort according to epoch id
    argsort = np.argsort(epoch_ids)
    epoch_ids = epoch_ids[argsort]
    val_avg_imps = val_avg_imps[argsort]
    test_avg_imps = test_avg_imps[argsort]

    return epoch_ids, val_avg_imps, test_avg_imps


if __name__ == '__main__':
    # @clock
    def snooze(seconds):
        time.sleep(seconds)


    # snooze(.123)

    path = '../data/test_res.txt'
    print(parse_res_table(path))
