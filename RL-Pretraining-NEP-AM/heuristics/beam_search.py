import os
import torch
import numpy as np
import networkx as nx
from copy import copy, deepcopy
from torch.utils.data.dataloader import DataLoader
from nets.attention_model import AttentionModel
from problems.network_enhancement_env import NetworkEnhancementEnv
from problems.network_generator import construct_network_seeds
from problems.network_dataset import NetworkDataset
from utils.functions import load_params, get_freer_gpu


# Model used for Beam Search
class BeamSearchModel(AttentionModel):
    def __init__(self,
                 node_dim,
                 embed_dim,
                 pos_enc,
                 feed_forward_hidden=512,
                 n_encode_layers=2,
                 n_heads=8,
                 tanh_clipping=10,
                 aggr_func=None,
                 device=torch.device("cpu")):
        super(BeamSearchModel, self).__init__(node_dim,
                                              embed_dim,
                                              pos_enc,
                                              feed_forward_hidden,
                                              n_encode_layers,
                                              n_heads,
                                              tanh_clipping,
                                              aggr_func,
                                              device)

    @torch.no_grad()
    def compute_log_p(self, env: NetworkEnhancementEnv):
        """
        TODO: MODIFY THIS FUNCTION for DIFFERENT MODELS.
        :param env:
        :return: log_p : (batch_size, graph_size)
                 graph_finished : (batch_size, )
        """
        graphs, first_nodes, node_banned, graph_finished = env.get_state()

        # Construct Statistical Features.
        statistical_features = [self.stat_feature_extractors[0](env.graphstates, graphs, self.node_dim),
                                self.stat_feature_extractors[1](graphs),
                                self.stat_feature_extractors[2](env.graphstates),
                                self.stat_feature_extractors[3](graphs)
                                ]

        # Extract relative_features only at odd steps
        if env.step_counter % 2 == 0:
            # (batch_size, graph_size, self.relative_feature_dim)
            relative_features = [torch.zeros(statistical_features[0].size(0),
                                             statistical_features[0].size(1),
                                             self.relative_feature_dim)]
        else:
            # Fix BUG : get_algebraic_distance would use the cache.
            self.statistical_features_cached = statistical_features
            relative_features = [extractor(graphs, first_nodes) for extractor in self.relative_feature_extractors]

        # (batch_size, graph_size, self.stat_feature_dim + self.relative_feature_dim)
        init_embed_ = torch.cat(statistical_features + relative_features, dim=-1).to(self.device)

        init_embed_ = self.init_embed(init_embed_)  # (batch_size, graph_size, embed_dim)
        embeds = self.embedder(init_embed_)  # (batch_size, graph_size, embed_dim)
        graph_embeds = self.get_graph_embed(embeds)  # (batch_size, embed_dim)

        # (batch_size, embed_dim)
        query = self._get_query(embeds, graph_embeds, first_nodes, graph_finished, env.step_counter)

        # (batch_size, graph_size, embed_dim)
        # --> (batch_size, graph_size, 3 * embed_dim)
        # --> (batch_size, graph_size, embed_dim) * 3
        glimpse_key, glimpse_val, logit_key = self.project_node_embeddings(embeds).chunk(3, dim=-1)

        mask = torch.BoolTensor(np.array(node_banned)).to(self.device)  # (batch_size, graph_size)

        # If a graph is finished, all of its nodes are banned.
        # However, this would produce the nan value and further cause the gradient to be nan.
        # For we do not care about the results of finished graph, set all of its nodes to be unbanned.
        graph_finished = torch.BoolTensor(graph_finished).to(self.device)
        mask[graph_finished] = False

        # (batch_size, graph_size)
        logits = self._calculate_logits(query, glimpse_key, glimpse_val, logit_key, mask)
        log_p = torch.log_softmax(logits, dim=-1)  # (batch_size, graph_size)

        return log_p, graph_finished

    @torch.no_grad()
    def rollout(self, env: NetworkEnhancementEnv):
        """
        :param env:
        :return:
        """

        # otherwise, env will be modified in-place by env.step
        env = deepcopy(env)

        # action_all_steps[i] : The List of Nodes Chosen for All Graphs in the i-th Step.
        # action_all_steps[i][j] : The Node Chosen for the j-th Graph in the i-th Step.
        action_all_steps = []

        while not env.is_terminal():
            _, selected, _ = self.act(env)
            actions = [u if u != -1 else None for u in selected.tolist()]
            env.step(actions)
            action_all_steps.append(actions)

        return action_all_steps


class TreeNode:
    def __init__(self, node, log_p):
        self.node = node
        self.log_p = log_p
        self.children = []
        self.idx_child_to_visit = 0

    def add_child(self, child):
        self.children.append(child)


def deep_first_search(root):
    """
    :param root: root
    :return: paths from root to leaves, each path contains (log prob of the path, [node_1, node_2, ...])
    """
    # TreeNode to Visit is on the top of the stack
    stack = [root]
    # nodes from bottom of the stack to the top of the stack
    prefix = [root.node]
    # log probability of a path : sum of log_p from bottom of the stack to the top of the stack
    log_p = root.log_p
    # paths from root to leaves, each element contains (log prob of the path, path)
    paths = []
    while stack:
        treenode: TreeNode = stack[-1]
        if treenode.idx_child_to_visit < len(treenode.children):
            child = treenode.children[treenode.idx_child_to_visit]
            treenode.idx_child_to_visit += 1
            # put child into the stack
            stack.append(child)
            prefix.append(child.node)
            log_p += child.log_p
        else:
            if len(treenode.children) == 0:
                paths.append((log_p, copy(prefix[1:])))
            # pop treenode from the stack
            stack.pop(-1)
            prefix.pop(-1)
            log_p -= treenode.log_p

    return paths


class BeamSearch:
    def __init__(self, model: BeamSearchModel, width):
        """
        :param model:
        :param width: Number of Children.
        """
        super(BeamSearch, self).__init__()
        self.model = model
        self.width = width

    def beam_search(self, env: NetworkEnhancementEnv):
        """
        :param env:
        :return: paths from root to leaves, each path contains (log prob of the path, [node_1, node_2, ...]).
        """
        assert len(env.graphstates) == 1, 'Environment Should Contain Only One Network.'

        # Bug Fix
        env = deepcopy(env)

        # The node/log_p for root is -1/0.0
        root = TreeNode(-1, 0.0)
        env_queue = []
        if not env.is_terminal():
            env_queue = [(root, env)]

        while env_queue:
            parent, env = env_queue.pop(0)
            # log_p : (1, graph_size)
            # graph_finished : (1, )
            log_p, graph_finished = self.model.compute_log_p(env)
            # (1, beam_width)
            weights, indices = torch.topk(log_p, k=self.width, dim=-1)

            # To exclude banned nodes.
            # Note : indices may include banned nodes. The weights of banned nodes are -inf.
            # weights and indices would be dimension 1 tensor.
            weights, indices = weights[weights != -torch.inf], indices[weights != -torch.inf]

            weights, indices = weights.tolist(), indices.tolist()

            children = [TreeNode(node, log_p) for node, log_p in zip(indices, weights)]

            for child in children:
                # Do not mess up the origin environment.
                child_env = deepcopy(env)
                child_env.step([child.node])
                # Construct the tree structure.
                parent.add_child(child)
                if not child_env.is_terminal():
                    env_queue.append((child, child_env))

        # Find paths from root to leaves, A path is a pair of (log probability, node list)
        return deep_first_search(root)


if __name__ == '__main__':
    model_saving_dir = '../outputs/BA_n_20_m_2/run_20221127T232037'
    checkpoint = 'epoch-2.pt'

    # Read Parameters from Model Saving Dir
    opts_ = load_params(model_saving_dir)

    # Set the device
    if opts_.use_cuda:
        opts_.device = torch.device(get_freer_gpu())
    else:
        opts_.device = torch.device("cpu")

    # Instantiate the Model
    model_ = BeamSearchModel(
        opts_.node_dim,
        opts_.embedding_dim,
        opts_.feed_forward_hidden,
        opts_.n_encode_layers,
        opts_.n_heads,
        opts_.tanh_clipping,
        opts_.device
    ).to(opts_.device)
    model_.set_decode_type("greedy")
    model_.eval()

    beam = BeamSearch(model_, width=3)

    checkpoint = torch.load(os.path.join(model_saving_dir, checkpoint))
    if 'model_state_dict' in checkpoint:
        model_.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_.load_state_dict(checkpoint)

    opts_.val_size = 1

    _, val_seeds, _ = construct_network_seeds(opts_.epoch_size, opts_.val_size, opts_.test_size)
    val_dataset = NetworkDataset(val_seeds, opts_)

    # Initialize DataLoader
    val_dataloader = DataLoader(val_dataset, opts_.val_size)

    for graphs in val_dataloader:
        val_graphs = [nx.from_numpy_array(graph.numpy()) for graph in graphs]
        env_ = NetworkEnhancementEnv(val_graphs,
                                     opts_.edge_budget_percentage)

        # Get An initial solution
        action_all_steps_ = model_.rollout(env_)
        actions_ = [list(item) for item in zip(*action_all_steps_)][0]

        env_.step([actions_[0]])
        # env_.step([actions_[1]])

        print(beam.beam_search(env_))
