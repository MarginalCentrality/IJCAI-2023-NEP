# Setting Up the Conda Environment
```shell
conda env create --file environment.yml
```

# Pretraining for Reinforcement Learning
Reinforcement Learning (RL) pretraining involves training models on small datasets and evaluating their performance on either in-distribution (ID) or out-of-distribution (OOD) instances. The directories RL-Pretraining-NEP-AM and RL-Pretraining-NEP-HAM are utilized for NEP-AM and NEP-HAM, respectively.

## Usage of RL-Pretraining-NEP-AM
The training process is conducted on Barabási-Albert (BA) networks and Erdős–Rényi (ER) networks for the targeted attack and random attack. Users have the flexibility to include different network generators or new attack methods.

The standard operational procedure is as follows:
1. Run the `run.py` script to pretrain the models. For instance,
```shell
python run.py
# Specify either BA or ER networks, for example, "BA_n_20_m_2" or "GNM_n_20_m_38".
# Note that the network instances will be generated automatically if this program is executed for the
# first time or if the generated networks are not stored.
--graph_model ${graph_model}
# Save the genenrated networks and prevent redundant network generation.
--store_graphs
# Define the root directory for storing the graphs.
--graph_storage_root ${graph_storage_root}
# Define the batch size for training and the size of the training set.
--batch_size 128
--epoch_size 16384
# Specify the batch size for validation and the size of the validation set.
--val_batch_size 128 
--val_size 128
# Determine the dimension of position encoding.
--node_dim 2
# Determine the dimension of node representation.
--embedding_dim 32
# Set the number of hidden units in the feedforward layer of the multi-head attention.
--feed_forward_hidden 512
# Specify the number of multi-head attention layers.
--n_encode_layers 3
# Determine the number of heads in each multi-head attention layer.
--n_head 8
# Define the coefficient used in the transformation of logits to balance exploitation and exploration.
--tanh_clipping 10
# Choose the aggregation function for generating graph representation, either "sum" or "mean".
--aggr_func sum
# Set the learning rate.
--lr_model 1e-4
# Specify the learning rate decay per epoch.
--lr_decay 1.0
# Define the number of epochs for training.
# In our experiment, for edge budget percentage of 1.0%/2.5%/5.0%, n_epochs is set as 3124/2344/1563.
--n_epochs 3124
# Use the seed to initialize the model parameters.
--seed 1234
# Set the maximum l2 norm for gradient clipping.
# If set to "inf", the gradient will not be clipped.
--max_grad_norm 1.0
# Choose the baseline method to reduce variance, another option is "exponential".
--baseline rollout
# Specify the coefficient for exponential moving average used in the exponential baseline.
--exp_beta 0.8
# Define the significance level in the t-test for updating the rollout baseline.
--bl_reject_threshold 0.05
# Choose the attack method, another option is "random_removal".
--method targeted_removal 
# Define parameters to control the Monte Carlo simulations.
# Please note that `sim_seed` and `n_mc_sims` specifically impact the random attack.
# In the case of the targeted attack, tie-breaking is determined by the node ID.
--sim_seed 42 
--n_mc_sims 40
# Control the number of edges to add to the network.
--edge_budget_percentage 1.0
# Scale the reward signal.
--scale_factor 100
# Specify the directory to store TensorFlow events.
--log_dir ${log_dir}
# Define the directory to save checkpoints.
--output_dir ${output_dir}
# Evaluate the model on the validation dataset every 10 batches to determine the best checkpoint.
--checkpoint_batches 10
# Back up the checkpoint every 100 epochs.
--checkpoint_epochs 100 
```

2. Employ `eval_on_val.py` to evaluate all saved checkpoints on the validation dataset. For instance,
```shell
python eval_on_val.py
# Specify the directory for saving the checkpoint, e.g., ${out_putdir}/${graph_model}/run_YYYYMMDDTHHMMSS.
--model_saving_dir ${model_saving_dir}
# Define the size of the validation set.
--test_size 128
```

3. Utilize `choose_checkpoints.py` to select either a standalone model or constuct an ensemble model from the saved checkpoints. For instance, 
```shell
python choose_checkpoints.py
--model_saving_dir ${model_saving_dir} 
# The "single" option selects the checkpoint with the highest validation performance.
# Other options like "cover_instances", "best_k", "last_k" can be employed to construct
# ensemble models from the saved checkpoints. 
--strategy single
# the number of independent models needed to construct an ensemble model.
# If the strategy is "single", this parameter would be overlooked. 
--k 5 
```

4. Evaluate the standalone or ensemble model on the ID test set by running `eval_on_test.py`.
```shell
python eval_on_test.py
--model_saving_dir ${model_saving_dir} # the directory of saving checkpoints, e.g., ${out_putdir}/${graph_model}/run_YYYYMMDDTHHMMSS
# Specify the checkpoints to use.
--strategy single
--k 5
# Define the size of the test set.
--test_size 128 
```

5. To evaluate the performance on the OOD test set, begin by executing `precompute_modifications.py` to identify the edges that need to be added to the network. Next, utilize `fast_eval_ood.py` to assess the model's performance on the OOD test set based on the identified edges.
```shell
python precompute_modifications.py
--model_saving_dir ${model_saving_dir} 
# Specify the model used to generate OOD instances.
--graph_model BA_n_500_m_2
# Set the maximum edge budget percentage for the OOD instances.
--largest_possible_ebp 0.2
# Define the number of OOD instances for reporting the OOD performance.
--test_size 128 
# Specify the checkpoints to use.
--strategy single
--k 5
# Set the number of processes running concurrently for precomputing modifications.
--n_process 1 

python fast_eval_ood.py
--model_saving_dir ${model_saving_dir}
# Specify the model for generating OOD instances.
--graph_model BA_n_500_m_2
# Define the number of edges to be inserted into the network, ensuring it is lower than the largest_possible_ebp. 
--edge_budget_percentage 0.1
# Set the maximum edge budget percentage for the OOD instances.
--largest_possible_ebp 0.2
# Define the number of instances used for reporting the OOD performance.
--test_size 128
# Specify the checkpoints to use.
--strategy single
--k 5
# Set the number of Monte Carlo simulations to estimate the robustness.
--n_mc_sims 200
# Set the number of concurrent processes for evaluating the robustness.
--n_process 8
```


6. Use `eval_local_search.py` or `eval_local_search_ensemble.py` to evaluate the performance of NCLS of a standalone or ensemble model. For instance, 
```shell
python eval_local_search.py
# Specify the number of instances for evaluating the NCLS performance.
--test_size 128
--model_saving_dir ${model_saving_dir}
# Choose the checkpoint for generating initial solutions and neighborhoods.
--checkpoint ${checkpoint}
# Select the method for initial solution generation, either "random_search" or "beam_search".
--init_sol_mtd random_search
# Control the generation of initial solutions under the random search method.
--random_search_seed 1234
# Define the beam width for initial solution search using beam search.
# Note: Avoid using a large beam width for long initial solutions.
--beam_width_init_sol 1
# Specify the number of initial solutions to be generated.
--n_init_sol 8
# Determine the number of parallel workers to execute NCLS.
--n_child_procs 8
# Specify the number of nodes to exchange. In our experiments, we used 1, 5, and 7.
--swap_size 5
# Set the sliding window step size.
--swap_step_size 1
# Define the beam width in the beam search for generating neighborhoods.
--beam_width 2
# Specify the upper limit of the neighborhood.
--nbor_size 100
# Optional. If enabled, NCLS is executed with a varying swap size.
--advanced_local_search
# Specify a list of swap sizes for the advanced local search.
--swap_size_list 7 5 1

# The usage of eval_local_search_ensemble.py is similar to eval_local_search.py.
# The main differences are the use of strategy and k parameters to select the checkpoints.
python eval_local_search_ensemble.py
# Specify the number of instances for evaluating the NCLS performance.
--test_size 128
--model_saving_dir ${model_saving_dir}
# Select the checkpoints for generating initial solutions and neighborhoods.
--strategy ${strategy}
--k 5
# Select the method for initial solution generation, either "random_search" or "beam_search".
--init_sol_mtd random_search
# Control the generation of initial solutions under the random search method.
--random_search_seed 1234
# Define the beam width for initial solution search using beam search.
# Note: Avoid using a large beam width for long initial solutions.
--beam_width_init_sol 1
# Specify the number of initial solutions to be generated.
--n_init_sol 8
# Determine the number of parallel workers to execute NCLS.
--n_child_procs 8
# Specify the number of nodes to exchange. In our experiments, we used 1, 5, and 7.
--swap_size 5
# Set the sliding window step size.
--swap_step_size 1
# Define the beam width in the beam search for generating neighborhoods.
--beam_width 2
# Specify the upper limit of the neighborhood.
--nbor_size 100
# Optional. If enabled, NCLS is executed with a varying swap size.
--advanced_local_search
# Specify a list of swap sizes for the advanced local search.
--swap_size_list 7 5 1
```

7. This step is optional for choosing a better standalone model. It has been observed that the validation performance tends to be noisy when the validation set is small. However, enlarging the validation set could significantly prolong the training process due to the frequent validation checks. A suggested approach is to assess only the top $k$ models that exhibit superior performance on the small validation set on the larger validation set to identify the most optimal model. To execute this procedure, the script `choose_checkpoints.py` is run with best_5 to select the top 5 models on the smaller validation dataset. Then the following command can be used to identify the best model:
```shell
python eval_on_val.py
# Specify the directory to store the checkpoint, for example, ${out_putdir}/${graph_model}/run_YYYYMMDDTHHMMSS.
--model_saving_dir ${model_saving_dir}
# Specify the size of the validation set.
--test_size 128
# Activate the use of an extra validation dataset.
--use_extra_val_set
# Determine the size and batch size of the extra validation dataset.
--extra_val_set_size 2048
--extra_val_set_batch_size 128
# Evaluate only the top 5 models on the smaller validation dataset.
--strategy best_k
--k 5
```


  

 



