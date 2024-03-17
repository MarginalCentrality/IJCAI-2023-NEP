import os

from tabulate import tabulate

from utils.functions import parse_res_table
import argparse
import numpy as np
from utils.choose_checkpoint import get_res_name
import re


def extract_imp_map(filepath):
    imp_map = {}
    with open(filepath, 'r') as in_file:
        for line in in_file:
            contents = line.strip().split()
            if '.pt' not in contents[0]:
                continue
            imp_map[contents[0]] = float(contents[2].split(':')[1])
    return imp_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze Results on the extra validation dataset~"
    )

    parser.add_argument('--network_model', type=str)
    parser.add_argument('--ood_network_model_list', nargs='*')
    parser.add_argument('--model_saving_dir', type=str, default='./', help="The Directory of Saving Models.")

    parser.add_argument('--strategy', type=str, choices=['cover_instances', 'best_k', 'consecutive_best_k', 'last_k',
                                                         'single'], default=None)
    parser.add_argument('--k', type=int, default=5, help="Number of checkpoints to choose.")

    args = parser.parse_args()

    strategy_config = get_res_name(args.strategy, args.k)

    extra_val_res_file_name = f"res_extra_val_{strategy_config}.txt"

    # Read contents  { epoch-xxx.pt: extra_val_imp, ... }
    extra_val_imp_map = {}
    with open(os.path.join(args.model_saving_dir, extra_val_res_file_name), 'r') as in_file:
        for line in in_file:
            chk_pt, extra_val_imp = line.strip().split()
            extra_val_imp_map[chk_pt] = float(extra_val_imp)

    # Get files Under the model_saving_dir
    files = os.listdir(args.model_saving_dir)
    # Exclude Directories
    files = [_ for _ in files if os.path.isfile(os.path.join(args.model_saving_dir, _))]

    # Read the Results of the ID Dataset
    re_id_val_imp = re.compile(f'res_{args.network_model}_[0-9T]+_{strategy_config}.txt')
    id_res_files = [_ for _ in files if re_id_val_imp.match(_) is not None]
    id_val_imp_map = extract_imp_map(os.path.join(args.model_saving_dir, id_res_files[0]))

    # Read the Results of the OOD Dataset
    ood_val_imp_map = {}
    for ood_network_model in args.ood_network_model_list:
        ood_val_imp_map[ood_network_model] = {}
        re_ood_val_imp = re.compile(f'res_{ood_network_model}_[0-9T]+_[0-9]+_{strategy_config}.txt')
        ood_res_files = [_ for _ in files if re_ood_val_imp.match(_) is not None]
        for ood_res_file in ood_res_files:
            edge_num = int(ood_res_file.split('_')[7])
            ood_val_imp_map[ood_network_model][edge_num] = extract_imp_map(os.path.join(args.model_saving_dir, ood_res_file))

    # Concatenate all Results
    res_table = []
    for chk, extra_val_imp in sorted(extra_val_imp_map.items(), key=lambda _:_[1], reverse=True):
        res_row = [chk, extra_val_imp, id_val_imp_map[chk]]

        def get_ood_network_size(ood_network_model_name):
            return int(ood_network_model_name.split('_')[2])

        for ood_network_model in sorted(ood_val_imp_map.keys(), key=get_ood_network_size):
            for edge_num in sorted(ood_val_imp_map[ood_network_model].keys()):
                res_row.append(ood_val_imp_map[ood_network_model][edge_num][chk])

        res_table.append(res_row)

    # Generate the Table
    table_header = ['checkpoint', 'Extra Val Imps', 'ID Val Imps', 'OOD-100', 'OOD-100', 'OOD-100', 'OOD-500', 'OOD-500', 'OOD-500']
    res_table = tabulate(res_table,
                         headers=table_header,
                         tablefmt='fancy_grid',
                         floatfmt=".4f",
                         showindex=True)

    print(res_table)








