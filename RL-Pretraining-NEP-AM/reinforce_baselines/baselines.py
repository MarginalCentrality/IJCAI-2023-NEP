from abc import abstractmethod

import torch
from torch.utils.data.dataloader import DataLoader
import networkx as nx
from scipy.stats import ttest_rel
from copy import deepcopy
from problems.network_dataset import EnhancedNetworkDataset
from problems.network_enhancement_env import NetworkEnhancementEnv
from problems.targeted_removal import TargetedRemoval
from problems.random_removal import RandomRemoval
from problems.graph_state import GraphState
from nets.attention_model import AttentionModel
from utils.functions import clock


class Baseline:
    @abstractmethod
    def eval(self, env: NetworkEnhancementEnv, rewards):
        pass

    def epoch_callback(self, model, epoch_id):
        pass

    def state_dict(self):
        return {}


class NoBaseLine(Baseline):
    def __init__(self):
        super(NoBaseLine, self).__init__()

    def eval(self, env: NetworkEnhancementEnv, rewards):
        return 0.0


class ExponentialBaseline(Baseline):
    def __init__(self, beta):
        super(ExponentialBaseline, self).__init__()
        self.beta = beta
        self.b_val = None

    @torch.no_grad()
    # @clock('ExponentialBaseline')
    def eval(self, env: NetworkEnhancementEnv, rewards):
        """
        :param env:
        :param rewards: (batch_size, )
        :return:
        """
        if self.b_val is None:
            self.b_val = rewards.mean()
        else:
            self.b_val = self.beta * self.b_val + (1 - self.beta) * rewards.mean()

        # (1, )
        return self.b_val

    def state_dict(self):
        return {
            'b_val': self.b_val
        }


class RolloutBaseline(Baseline):
    def __init__(self,
                 model: AttentionModel,
                 init_seed,
                 opts,
                 ):
        """
        :param model:
        :param init_seed:
        :param opts:
                -- graph_model
                -- store_graphs
                -- graph_storage_root
                -- val_size
                -- val_batch_size
                -- method
                -- edge_budget_percentage
                -- n_mc_sims
                -- bl_reject_threshold
                -- device
        """
        super(RolloutBaseline, self).__init__()
        self.model = None
        self.seed_to_use = init_seed
        self.opts = opts
        self.dataloader = None

        self.method = None
        if self.opts.method == 'random_removal':
            self.method = RandomRemoval
        elif self.opts.method == 'targeted_removal':
            self.method = TargetedRemoval
        else:
            raise NotImplementedError(f"{self.opts.method} has not been implemented.")

        # Validations of self.model on self.dataloader
        self.vals = None
        self.mean = None  # Mean of self.vals

        # Cache the robustness estimation for the initial network
        self.init_robustness_estimation = {}

        self.imps_cache = {}
        self._update_model(model)

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

    def _update_model(self, model):
        """
        :param model:
        :return:
        """
        self.model = deepcopy(model)
        self.model.set_decode_type("greedy")
        self.model.eval()

        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        baseline_dataset = EnhancedNetworkDataset(
            list(range(self.seed_to_use, self.seed_to_use + self.opts.val_size)),
            self.opts
        )
        self.seed_to_use = self.seed_to_use + self.opts.val_size

        self.dataloader = DataLoader(baseline_dataset, self.opts.val_batch_size)

        self.vals = self.eval_model(self.model)
        self.mean = self.vals.mean()

        # Clear the Cache
        self.imps_cache = {}

    @torch.no_grad()
    def eval_model(self, model):
        vals = []
        for graphs, seeds in self.dataloader:
            # numpy_array ----> networkx obj
            graphs = [nx.from_numpy_array(graph.numpy()) for graph in graphs]

            # Construct Graph States
            graphstates = [GraphState(g,
                                      self.opts.edge_budget_percentage,
                                      seed=seeds[idx])
                           for idx, g in enumerate(graphs)]

            env = NetworkEnhancementEnv(graphstates)

            # (batch_size, )
            # (batch_size, )
            init_robustness_estimation = torch.FloatTensor(self.get_init_robustness_estimation(env)).to(
                self.opts.device)

            # finished_env : finished environment
            _, _, _, finished_env = model(env)

            # (batch_size, )
            final_robustness_estimation = torch.FloatTensor(finished_env.
                                                            calculate_robustness(self.method,
                                                                                 self.opts.n_mc_sims,
                                                                                 self.opts.sim_seed,
                                                                                 self.opts.reuse_hash)) \
                .to(self.opts.device)

            # (batch_size, )
            vals.append(final_robustness_estimation - init_robustness_estimation)

        return torch.cat(vals)

    @torch.no_grad()
    # @clock('RolloutBaseline')
    def eval(self, env: NetworkEnhancementEnv, rewards):

        # Get seed of each graph in env
        seeds = [g_state.seed.item() for g_state in env.graphstates]

        # Check the Cache
        imps = [-torch.inf] * env.batch_size
        idx_to_build = []
        for idx, seed in enumerate(seeds):
            imp = self.imps_cache.get(seed, None)
            if imp is None:
                idx_to_build.append(idx)
            else:
                imps[idx] = imp

        imps = torch.FloatTensor(imps)

        # There exist some graphs to be evaluated.
        if len(idx_to_build) > 0:
            env = NetworkEnhancementEnv([env.graphstates[idx] for idx in idx_to_build])

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

            imps[imps == -torch.inf] = final_robustness_estimation - init_robustness_estimation

            # Update the Cache
            for idx in idx_to_build:
                self.imps_cache[seeds[idx]] = imps[idx].item()

        # (batch_size, )
        imps = imps.to(self.opts.device)
        return imps

    @torch.no_grad()
    # @clock('RolloutBaseline')
    def epoch_callback(self, model, epoch_id):
        """
        :param model:
        :param epoch_id:
        :return:
        """

        model.set_decode_type("greedy")  # TO RECOVER
        model.eval()  # TO RECOVER

        candidate_vals = self.eval_model(model)
        candidate_mean = candidate_vals.mean()

        model.set_decode_type("sampling")  # RECOVER
        model.train()  # RECOVER

        if candidate_mean > self.mean:
            # Test for the null hypothesis that two related samples have identical average (expected) values.
            t, p = ttest_rel(candidate_vals.cpu(), self.vals.cpu())
            # assert t > 0, "t-statistic should be positive"
            p_val = p / 2.0  # one-sided
            print(f'p-value: {p_val}')
            if p_val < self.opts.bl_reject_threshold:  # Reject the null hypothesis
                print("Update baseline")
                self._update_model(model)

    def state_dict(self):
        return {
            'model_state_dict': self.model.state_dict(),
            # Note : When recovering the baseline, the _update_model() would be invoked.
            # Thus, we need to recover the seed_to_use.
            'seed_to_use': self.seed_to_use - self.opts.val_size
        }


class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        """
        :param baseline: baseline to warm up.
        :param n_epochs: number of epochs to warm up.
        :param warmup_exp_beta: decay in exponential baseline.
        """
        super(WarmupBaseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warm up must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.n_epochs = n_epochs
        self.n_callbacks = 0

    def eval(self, env: NetworkEnhancementEnv, rewards):

        if self.n_callbacks < self.n_epochs:
            return self.warmup_baseline.eval(env, rewards)  # (1, )
        else:
            return self.baseline.eval(env, rewards)  # (batch_size, )

    def epoch_callback(self, model, epoch_id):
        # Need to call epoch callback of baseline (even we have not used it)
        self.baseline.epoch_callback(model, epoch_id)
        self.n_callbacks = self.n_callbacks + 1

    def state_dict(self):
        return {
            'baseline': self.baseline.state_dict(),
            'n_callbacks': self.n_callbacks,
            'b_val': self.warmup_baseline.b_val
        }
