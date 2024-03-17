import math
import os
import torch
import time
from datetime import timedelta
import networkx as nx
import torch.optim as optim
import torch.nn.utils as utils
from torch.utils.data import dataloader
from tensorboard_logger import Logger as TbLogger
from nets.attention_model import AttentionModel
from tqdm import tqdm
from problems.network_enhancement_env import NetworkEnhancementEnv
from problems.random_removal import RandomRemoval
from problems.targeted_removal import TargetedRemoval
from problems.graph_state import GraphState
from reinforce_baselines.baselines import Baseline, NoBaseLine, ExponentialBaseline, RolloutBaseline

from utils.functions import clock


class REINFORCE:
    def __init__(self,
                 model: AttentionModel,
                 baseline: Baseline,
                 opts,
                 tb_logger: TbLogger = None
                 ):
        """
        :param model:
        :param baseline:
        :param tb_logger: TbLogger
        :param opts:
           - lr_model
           - lr_decay
           - method
           - percentage
           - num_mc_sims
           - max_grad_norm
           - no_tensor_board
           - no_progress_bar
           - run_name
           - epoch_size
           - n_epochs
           - batch_size
           - validation_change_threshold
           - checkpoint_batches
           - checkpoint_epochs
           - save_dir
           - scale_factor
        """
        self.model = model
        self.baseline = baseline
        self.tb_logger = tb_logger
        self.opts = opts

        # Initialize Optimizer
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.opts.lr_model
        )

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lr_lambda=lambda epoch: self.opts.lr_decay ** epoch)

        self.method = None
        if self.opts.method == 'random_removal':
            self.method = RandomRemoval
        elif self.opts.method == 'targeted_removal':
            self.method = TargetedRemoval
        else:
            raise NotImplementedError(f"{self.opts.method} has not been implemented.")

        # best average robustness improvements on validation dataset
        self.best_val = 0.0

        # Cache the robustness estimation for the initial network
        self.init_robustness_estimation = {}

    def get_init_robustness_estimation(self, env: NetworkEnhancementEnv):
        # Get seed of each graph in env
        seeds = [g_state.seed.item() for g_state in env.graphstates]

        # Check the Cache
        robustness_list = [None] * env.batch_size
        idx_to_estimate = []
        for idx, seed in enumerate(seeds):
            robustness = self.init_robustness_estimation.get(seed, None)
            if robustness is None:
                idx_to_estimate.append(idx)
            else:
                robustness_list[idx] = robustness

        # There exist some graphs to estimate there robustness.
        if len(idx_to_estimate) > 0:
            env = NetworkEnhancementEnv([env.graphstates[idx] for idx in idx_to_estimate])

            init_robustness_estimation = env.calculate_robustness(self.method,
                                                                  self.opts.n_mc_sims,
                                                                  self.opts.sim_seed,
                                                                  self.opts.reuse_hash)

            # Update the Cache
            for i, idx in enumerate(idx_to_estimate):
                self.init_robustness_estimation[seeds[idx]] = init_robustness_estimation[i]
                robustness_list[idx] = init_robustness_estimation[i]

        return robustness_list

    # @clock('REINFORCE')
    def train_batch(self, graphstates, batch_id):
        """
        :param graphstates: A list of GraphState
        :param batch_id: Batch id in the whole training process rather than in the epoch.
        :return:
        """
        env = NetworkEnhancementEnv(graphstates)

        # (batch_size, )
        init_robustness_estimation = torch.FloatTensor(self.get_init_robustness_estimation(env)).to(self.opts.device)

        # log_p_all_steps : (batch_size, step_num, graph_size)
        # selected_all_steps : (batch_size, step_num)
        # graph_finished_all_steps : (batch_size, step_num)
        # finished_env : finished environment
        log_p_all_steps, selected_all_steps, graph_finished_all_steps, finished_env = self.model(env)

        # (batch_size, step_num, graph_size) ----> (batch_size, step_num, 1) ---> (batch_size, step_num)
        selected_all_steps[selected_all_steps == -1] = 0  # function gather cannot be indexed with -1

        log_p_all_steps = torch.gather(log_p_all_steps,
                                       -1,
                                       selected_all_steps[:, :, None]).squeeze(-1)  # (batch_size, step_num)

        # log_p_all_steps[graph_finished_all_steps] = 0.0  # To deal with finished graphs.

        log_p_all_steps = log_p_all_steps.masked_fill(graph_finished_all_steps, 0.0)

        # (batch_size, )
        final_robustness_estimation = torch.FloatTensor(finished_env.
                                                        calculate_robustness(self.method,
                                                                             self.opts.n_mc_sims,
                                                                             self.opts.sim_seed,
                                                                             self.opts.reuse_hash)).to(self.opts.device)

        # Compute rewards : (batch_size, )
        rewards = final_robustness_estimation - init_robustness_estimation

        bl_val = self.baseline.eval(env, rewards)  # (batch_size, ) or (1, )

        # Compute loss function
        # IMPORTANT : DO NOT MISS THE MINUS SIGN.
        reinforce_loss = (-1 * self.opts.scale_factor * (rewards - bl_val) * log_p_all_steps.sum(-1)).mean()

        # Backward process
        self.optimizer.zero_grad()
        reinforce_loss.backward()
        grad_norm = utils.clip_grad_norm_(self.model.parameters(), self.opts.max_grad_norm)
        self.optimizer.step()

        # Log Info
        grad_norm_clipped = min(grad_norm, self.opts.max_grad_norm)

        # Log values to screen
        # TODO: why avg_imp can be negative ?
        avg_imp = rewards.mean(dim=0).item()
        print("batch id: {}, avg imp: {:.3f}".format(batch_id, avg_imp))
        print("grad_norm: {:.3f}, clipped: {:.3f}".format(grad_norm, grad_norm_clipped))

        # Log values to tensorboard
        if not self.opts.no_tensorboard:
            self.tb_logger.log_value('avg_imp', avg_imp, batch_id)
            self.tb_logger.log_value('actor_loss', reinforce_loss.item(), batch_id)
            self.tb_logger.log_value('nll', -1 * log_p_all_steps.mean().item(), batch_id)
            self.tb_logger.log_value('grad_norm', grad_norm, batch_id)
            self.tb_logger.log_value('grad_norm_clipped', grad_norm_clipped, batch_id)

        return rewards.mean().item(), reinforce_loss.item()

    @torch.no_grad()
    def validate_batch(self, graphstates):
        """
        :param graphstates: A list of graph state
        :return:
        """
        env = NetworkEnhancementEnv(graphstates)

        # (batch_size, )
        init_robustness_estimation = torch.FloatTensor(self.get_init_robustness_estimation(env))

        # finished_env : finished environment
        _, _, _, finished_env = self.model(env)

        # (batch_size, )
        final_robustness_estimation = torch.FloatTensor(finished_env.
                                                        calculate_robustness(self.method,
                                                                             self.opts.n_mc_sims,
                                                                             self.opts.sim_seed,
                                                                             self.opts.reuse_hash))

        # (batch_size, )
        return final_robustness_estimation - init_robustness_estimation

    @torch.no_grad()
    # @clock('REINFORCE')
    def validate(self, val_dataloader: dataloader):
        # TO RECOVER self.model
        self.model.set_decode_type("greedy")
        self.model.eval()

        improvements = []
        for graphs, seeds in tqdm(val_dataloader, disable=self.opts.no_progress_bar):
            # numpy array ----> networkx object
            graphs = [nx.from_numpy_array(graph.numpy()) for graph in graphs]

            # Construct graph states
            # Construct Graph States
            graphstates = [GraphState(g,
                                      self.opts.edge_budget_percentage,
                                      seed=seeds[idx])
                           for idx, g in enumerate(graphs)]
            improvements.append(self.validate_batch(graphstates))

        # RECOVER self.model
        self.model.set_decode_type('sampling')
        self.model.train()

        # (len(val_data), )
        improvements = torch.cat(improvements, dim=0)
        avg_imp = improvements.mean()

        # Do not forget to divide by the size of the data.
        # We are computing the standard deviation of avg_imp.
        print("Val imps: {} +- {}".format(
            avg_imp,
            torch.std(improvements) / math.sqrt(improvements.size(0))))

        return avg_imp

    # @clock('REINFORCE')
    def train_epoch(self, epoch_id, training_dataloader: dataloader, val_dataloader: dataloader):
        print("Start train epoch {}, lr={} for run {}".format(epoch_id,
                                                              self.optimizer.param_groups[0]['lr'],
                                                              self.opts.run_name))

        # batch id in the whole training process rather than in the epoch
        batch_id = epoch_id * (self.opts.epoch_size // self.opts.batch_size)

        start_time = time.time()

        if not self.opts.no_tensorboard:
            self.tb_logger.log_value('learnrate_pg0', self.optimizer.param_groups[0]['lr'], batch_id)

        self.model.train()
        self.model.set_decode_type('sampling')

        for graphs, seeds in tqdm(training_dataloader, disable=self.opts.no_progress_bar):

            # numpy array ----> networkx object
            graphs = [nx.from_numpy_array(graph.numpy()) for graph in graphs]

            # Construct Graph States 
            graphstates = [GraphState(g,
                                      self.opts.edge_budget_percentage,
                                      seed=seeds[idx])
                           for idx, g in enumerate(graphs)]

            self.train_batch(graphstates, batch_id)

            if self.opts.checkpoint_batches != 0 and batch_id % self.opts.checkpoint_batches == 0:
                avg_imp = self.validate(val_dataloader)
                if not self.opts.no_tensorboard:
                    self.tb_logger.log_value("val_avg_imp", avg_imp, batch_id)

                if avg_imp - self.best_val > 0.0:
                    print(f"rejoice! found a better validation loss at batch {batch_id}.")
                    self.best_val = avg_imp
                    print("Saving model and state...")
                    torch.save(
                        {'epoch_id': epoch_id, 'model_state_dict': self.model.state_dict()},
                        os.path.join(self.opts.save_dir, f"{self.opts.run_name}.pt")
                    )

            batch_id += 1

        epoch_duration = time.time() - start_time  # seconds
        print("Finished epoch {}, took {}".format(epoch_id,
                                                  str(timedelta(seconds=epoch_duration))
                                                  ))

        self.baseline.epoch_callback(self.model, epoch_id)

        # learning rate scheduler should be called at end of each epoch
        self.scheduler.step()

        # Save a General Checkpoint
        if (self.opts.checkpoint_epochs != 0 and epoch_id % self.opts.checkpoint_epochs == 0) \
                or epoch_id == self.opts.n_epochs - 1:
            torch.save(
                {
                    'epoch_id': epoch_id,
                    'best_val': self.best_val,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'baseline_state_dict': self.baseline.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all()
                },
                os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch_id))
            )


if __name__ == '__main__':
    pass
    # import argparse
    # import networkx as nx
    #
    # from torch.utils.data import Dataset
    # from torch.utils.data.dataloader import DataLoader
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--lr_model', type=float)
    # parser.add_argument('--lr_decay', type=float)
    # parser.add_argument('--method', type=str)
    # parser.add_argument('--percentage', type=int)
    # parser.add_argument('--num_mc_sims', type=int)
    # parser.add_argument('--max_grad_norm', type=float)
    # parser.add_argument('--no_tensorboard', action='store_true')
    # parser.add_argument('--no_progress_bar', action='store_true')
    # parser.add_argument('--run_name', default='test_reinforce')
    # parser.add_argument('--epoch_size', type=int)
    # parser.add_argument('--batch_size', type=int)
    # parser.add_argument('--validation_change_threshold', type=float)
    # parser.add_argument('--checkpoint_batches', type=int)
    # parser.add_argument('--save_dir', type=str)
    #
    #
    # class TestDataset(Dataset):
    #     def __init__(self, graphs):
    #         self.data = graphs
    #
    #     def __getitem__(self, idx):
    #         return self.data[idx]
    #
    #     def __len__(self):
    #         return len(self.data)

    # # Case 1
    # # ---- Test REINFORCE  ----
    # e_list_1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # # elist1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    # e_list_2 = [(0, 1), (0, 2), (0, 3)]
    # e_list_3 = [(0, 1), (0, 3), (2, 3)]
    # graphs_ = [nx.Graph(e_list) for e_list in [e_list_1, e_list_2, e_list_3]]
    # test = TestDataset(graphs_)
    # test_dataloader = DataLoader(test, batch_size=len(graphs_))
    #
    # node_dim_ = 2
    # embed_dim_ = 16
    # attn = AttentionModel(node_dim_, embed_dim_)
    # attn.set_decode_type('sampling')
    # opts = parser.parse_args([
    #     '--lr_model', '1e-4',
    #     '--lr_decay', '1.0',
    #     '--method', 'targeted_removal',
    #     '--percentage', '20',
    #     '--num_mc_sims', '1',
    #     '--max_grad_norm', '1.0',
    #     '--no_tensorboard',
    #     '--epoch_size', '3',
    #     '--batch_size', '3',
    #     '--validation_change_threshold', '0.0',
    #     '--checkpoint_batches', '20',
    #     '--save_dir', './test_reinforce'
    # ])
    #
    # baseline = NoBaseLine()
    # reinforce = REINFORCE(attn, baseline, opts)
    # # torch.autograd.set_detect_anomaly(True)
    # reinforce.train_batch(graphs_, batch_id=0)

    # # Case 2
    # # ---- Test Robustness Improvement Line Graph  ----
    # #   percentage    edge_num     ttl_batch_num    reward = robustness improvement (targeted attack)    final
    # #   10            2            500              0.17                                                  0.33
    # #   20            3            5000             0.83                                                  1
    # #   25            4            5000             0.83 (max_grad_norm=math.inf)                         1
    # #   40            6            5000             0.83 (max_grad_norm=math.inf)                         1
    # # from tensorboard_logger import Logger as TbLogger
    # # tb_logger = TbLogger('./enhance_line_graph')
    #
    # e_list_1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    # graphs_ = [nx.Graph(e_list) for e_list in [e_list_1]]
    # node_dim_ = 2
    # embed_dim_ = 32
    #
    # torch.manual_seed(1234)
    # attn = AttentionModel(node_dim_, embed_dim_)
    # attn.set_decode_type('sampling')
    #
    # opts = parser.parse_args([
    #     '--lr_model', '1e-4',
    #     '--lr_decay', '1.0',
    #     '--method', 'targeted_removal',
    #     '--percentage', '20',
    #     '--num_mc_sims', '1',
    #     '--max_grad_norm', 'inf',
    #     '--no_tensorboard',
    #     '--epoch_size', '3',
    #     '--batch_size', '3',
    #     '--validation_change_threshold', '0.0',
    #     '--checkpoint_batches', '20',
    #     '--save_dir', './test_reinforce'
    # ])
    # ttl_batch_num = 1000
    #
    # baseline_ = NoBaseLine()
    #
    # reinforce = REINFORCE(attn, baseline_, opts)
    # # torch.autograd.set_detect_anomaly(True)
    #
    # for i in range(ttl_batch_num):
    #     reward_, reinforce_loss_ = reinforce.train_batch(graphs_, i)
    #     print("batch id:{}, reward:{:.3f}, loss:{:.3f}".format(i, reward_, reinforce_loss_))

    # Case 3
    # ---- Test Robustness Improvement with Tree Graph  ----
    #   percentage    edge_num     ttl_batch_num    reward = robustness improvement (targeted attack)    final
    # from tensorboard_logger import Logger as TbLogger
    # tb_logger = TbLogger('./enhance_binary_tree')
    # e_list_1 = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    # graphs_ = [nx.Graph(e_list_1)]
    # node_dim_ = 2
    # embed_dim_ = 32
    #
    # torch.manual_seed(1234)
    # attn = AttentionModel(node_dim_, embed_dim_)
    # attn.set_decode_type('sampling')
    #
    # beta_ = 0.95
    # baseline = ExponentialBaseline(beta_)
    #
    # opts = parser.parse_args([
    #     '--lr_model', '1e-4',
    #     '--lr_decay', '1.0',
    #     '--method', 'targeted_removal',
    #     '--percentage', '20',
    #     '--num_mc_sims', '1',
    #     '--max_grad_norm', 'inf',
    #     '--no_tensorboard',
    #     '--epoch_size', '3',
    #     '--batch_size', '3',
    #     '--validation_change_threshold', '0.0',
    #     '--checkpoint_batches', '20',
    #     '--save_dir', './test_reinforce'
    # ])
    #
    # ttl_batch_num = 5000
    #
    # reinforce = REINFORCE(attn, baseline, opts)
    # # torch.autograd.set_detect_anomaly(True)
    #
    # for i in range(ttl_batch_num):
    #     reward_, reinforce_loss_ = reinforce.train_batch(graphs_, i)
    #     # tb_logger.log_value("reward", reward_, i)
    #     print("batch id:{}, reward:{:.3f}, loss:{:.3f}".format(i, reward_, reinforce_loss_))

    # # Case 4
    # # ---- Gradient is not infected by a finished network ----
    # e_list_1 = [(0, 1), (0, 2), (0, 3)]
    # e_list_2 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # graphs_ = [nx.Graph(e_list) for e_list in [e_list_1, e_list_2]]
    #
    # node_dim_ = 2
    # embed_dim_ = 16
    # torch.manual_seed(1234)
    # attn = AttentionModel(node_dim_, embed_dim_)
    # attn.set_decode_type('greedy')
    # opts = parser.parse_args([
    #     '--lr_model', '1e-2',
    #     '--lr_decay', '1.0',
    #     '--method', 'targeted_removal',
    #     '--percentage', '20',
    #     '--num_mc_sims', '1',
    #     '--max_grad_norm', '1.0',
    #     '--no_tensorboard',
    #     '--epoch_size', '3',
    #     '--batch_size', '3',
    #     '--validation_change_threshold', '0.0',
    #     '--checkpoint_batches', '20',
    #     '--save_dir', './test_reinforce'
    # ])
    # ttl_batch_num = 100
    # torch.set_printoptions(precision=4, sci_mode=True)
    # baseline = NoBaseLine()
    #
    # reinforce = REINFORCE(attn, baseline, opts)
    # for i in range(ttl_batch_num):
    #     print('------------- BEGIN -------------')
    #     # Add code to print gradient of some parameter in train_batch
    #     reward_, reinforce_loss_ = reinforce.train_batch(graphs_)
    #     print('------------- END -------------')
    #     print('\n')

    # Case 5
    # ---- Test Robustness Improvement with Exponential BaseLine  ----
    # e_list_1 = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    # graphs_ = [nx.Graph(e_list) for e_list in [e_list_1, e_list_1, e_list_1]]
    # node_dim_ = 2
    # embed_dim_ = 32
    # beta_ = 0.95
    # baseline = ExponentialBaseline(beta_)
    # torch.manual_seed(1234)
    # attn = AttentionModel(node_dim_, embed_dim_)
    # attn.set_decode_type('sampling')
    # opts = parser.parse_args([
    #     '--lr_model', '1e-4',
    #     '--lr_decay', '1.0',
    #     '--method', 'targeted_removal',
    #     '--percentage', '20',
    #     '--num_mc_sims', '1',
    #     '--max_grad_norm', 'inf',
    #     '--no_tensorboard',
    #     '--epoch_size', '3',
    #     '--batch_size', '3',
    #     '--validation_change_threshold', '0.0',
    #     '--checkpoint_batches', '20',
    #     '--save_dir', './test_reinforce'
    # ])
    #
    # ttl_batch_num = 100
    # reinforce = REINFORCE(attn, baseline, opts)
    # # torch.autograd.set_detect_anomaly(True)
    #
    # for i in range(ttl_batch_num):
    #     reward_, reinforce_loss_ = reinforce.train_batch(graphs_, i)
    #     print("batch id:{}, reward:{:.3f}, loss:{:.3f}".format(i, reward_, reinforce_loss_))
