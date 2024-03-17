from utils.functions import parse_res_table
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze Random Attack Results~"
    )

    parser.add_argument('--res_table_path', type=str, default='./', help="The path of result table.")
    parser.add_argument('--k', type=int, default=10, help="Find k consecutive checkpoints")
    parser.add_argument('--col_id_epoch_id', type=int, default=3, help="The column id of epoch_id")
    parser.add_argument('--col_id_val_avg_imp', type=int, default=4, help="The column id of val_avg_imp")
    parser.add_argument('--col_id_test_avg_imp', type=int, default=7, help="The column id of test_avg_imp")
    parser.add_argument('--splitter', type=str, default='│')

    args = parser.parse_args()

    epoch_ids, val_avg_imps, test_avg_imps = parse_res_table(args.res_table_path,
                                                             args.col_id_epoch_id,
                                                             args.col_id_val_avg_imp,
                                                             args.col_id_test_avg_imp,
                                                             args.splitter)
    l = len(val_avg_imps)

    assert args.k <= l, 'k should be less than or equal to l'

    partial_sum = []
    for s in range(0, l-args.k+1):
        partial_sum.append(np.sum(val_avg_imps[s: s+args.k]))

    partial_sum = np.array(partial_sum, dtype=float)

    idx = np.argmax(partial_sum)

    epoch_ids = epoch_ids[idx: idx+args.k]
    val_avg_imps = val_avg_imps[idx: idx+args.k]
    test_avg_imps = test_avg_imps[idx: idx + args.k]

    print("The consecutive epoch ids : ")
    print(epoch_ids)
    print("The consecutive val avg imps : ")
    print(val_avg_imps)
    print("The consecutive test avg imps : ")
    print(test_avg_imps)
    print("The avg test imp : {0:.3f}".format(np.mean(test_avg_imps)))
    print("The max test imp : {0:.3f}".format(np.max(test_avg_imps)))
    print("The max test imp arrives at {0:d}-th epoch".format( epoch_ids[np.argmax(test_avg_imps)] ))










