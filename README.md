# Reinforcement Learning Pretraining
Reinforcement Learning (RL) pretraining involves training models on small datasets and assessing their performance on instances sampled from in-distribution (ID) or  out-of-distribution (OOD). The directories RL-Pretraining-NEP-AM and RL-Pretraining-NEP-HAM are utilized for NEP-AM and NEP-HAM, respectively.

## Useage of RL-Pretraining-NEP-AM
We realize the training process on Barabási-Albert (BA) networks and Erdős–Rényi (ER) networks for targeted and random attacks. For training on different network types or attack strategies, users can easily incorporate network generators or new attack methods.

The standard operational sequence is outlined as follows:
  1. Execute run.py to pretrain the models.
  2. Utilize eval_on_val.py to assess all stored checkpoints on the validation dataset.
  3. Employ choose_checkpoints.py to select either the standalone model or construct an ensemble model from the stored checkpoints.
  4. Run eval_on_test.py to evaluate the standalone or ensemble model on the ID test set.
  5. For evaluating performance on the OOD test set, execute precompute_modifications.py to determine the edges to add to the network. Subsequently, use fast_eval_ood.py to evaluate the model's performance on the OOD test set based on the edges identified earlier.
  6. Execute eval_local_search.py/eval_local_search_ensemble.py to evaluate the performance of NCLS of a standalone/ensemble model.
 



