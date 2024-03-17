# Setting Up the Conda Environment
```shell
conda env create --file environment.yml
```

# Pretraining for Reinforcement Learning
Reinforcement Learning (RL) pretraining involves training models on small datasets and evaluating their performance on either in-distribution (ID) or out-of-distribution (OOD) instances. The directories RL-Pretraining-NEP-AM and RL-Pretraining-NEP-HAM are utilized for NEP-AM and NEP-HAM, respectively.

## Usage of RL-Pretraining-NEP-AM
The training process is conducted on Barabási-Albert (BA) networks and Erdős–Rényi (ER) networks for the targeted attack and random attack. Users have the flexibility to include different network generators or new attack methods.

The standard operational procedure is as follows:
1. Run the `run.py` script to pretrain the models.
2. Employ `eval_on_val.py` to evaluate all saved checkpoints on the validation dataset.
3. Use `choose_checkpoints.py` to choose a standalone model or construct an ensemble model from the saved checkpoints.
4. Evaluate the standalone or ensemble model on the ID test set by running `eval_on_test.py`.
5. To assess performance on the OOD test set, run `precompute_modifications.py` to identify the edges to add to the network. Then, utilize `fast_eval_ood.py` to evaluate the model's performance on the OOD test set based on the identified edges.
6. Use `eval_local_search.py` or `eval_local_search_ensemble.py` to evaluate the performance of NCLS of a standalone or ensemble model.
