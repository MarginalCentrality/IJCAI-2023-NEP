import math

import torch
from torch import nn
from torch.distributions.categorical import Categorical
import networkx as nx
import numpy as np
import numpy.linalg as linalg
from math import sqrt
from copy import deepcopy
from nets.graph_encoder import GraphAttentionEncoder
from problems.network_enhancement_env import NetworkEnhancementEnv
from utils.node_feature_extractor import NodeFeatureExtractor
from utils.functions import clock


class AttentionModel(nn.Module):

    def __init__(self,
                 node_dim,
                 embed_dim,
                 pos_enc,
                 feed_forward_hidden=512,
                 n_encode_layers=2,
                 n_heads=8,
                 tanh_clipping=10,
                 aggr_func=None,
                 device=torch.device("cpu"),
                 to_double=False
                 ):
        super(AttentionModel, self).__init__()

        assert embed_dim % n_heads == 0

        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.pos_enc = pos_enc
        self.feed_forward_hidden = feed_forward_hidden
        self.n_encode_layers = n_encode_layers
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.aggr_func = aggr_func
        self.device = device
        self.to_double = to_double

        self.W_placeholder = nn.Parameter(torch.Tensor(embed_dim))
        self.W_placeholder.data.uniform_(-1, 1)

        assert pos_enc == 'dynamic' or pos_enc == 'fix' or pos_enc == 'approx', \
            f'{pos_enc} has not been implememted~'

        if pos_enc == 'dynamic':
            pos_enc_extractor = NodeFeatureExtractor.get_shallow_embed
        elif pos_enc == 'fix':
            pos_enc_extractor = NodeFeatureExtractor.get_fixed_shallow_embed
        elif pos_enc == 'approx':
            pos_enc_extractor = NodeFeatureExtractor.approx_shallow_embed

        # Statistical Features
        # - shallow embedding  : self.node_dim
        # - normalized degree : 1
        # - clustering coefficient : 1
        # - avg neighbor degree : 1
        self.stat_feature_extractors = [
            pos_enc_extractor,
            NodeFeatureExtractor.get_degree_centrality,
            NodeFeatureExtractor.get_clustering,
            NodeFeatureExtractor.get_average_neighbor_degree
        ]
        self.stat_feature_dim = len(self.stat_feature_extractors) + self.node_dim - 1

        # Relative Features to the first chosen node
        # - normalized distance : 1
        # - normalized degree product : 1
        # - normalized algebraic distance : 1
        # - jaccard_coefficient : 1
        self.relative_feature_extractors = [
            NodeFeatureExtractor.get_distance,
            NodeFeatureExtractor.get_degree_product,
            self.get_algebraic_distance,
            NodeFeatureExtractor.get_jaccard_coefficient
        ]
        self.relative_feature_dim = len(self.relative_feature_extractors)

        self.init_embed = nn.Linear(self.stat_feature_dim + self.relative_feature_dim, self.embed_dim)

        self.embedder = GraphAttentionEncoder(
            self.n_heads,
            self.embed_dim,
            self.n_encode_layers,
            self.feed_forward_hidden
        )

        # For each node we compute (key to compute glimpse, value to compute glimpse, key to compute logit)
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        # Step context consists of global information and local information
        self.project_step_context = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        # Nore n_head * val_dim == embed_dim so input to project_out is embed_dim
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)

        self.decode_type = None  # greedy or sampling

    def get_algebraic_distance(self, graphs, sources):
        """
        :param graphs: useless parameter
        :param sources:
        :return:
        """
        # (batch_size, graph_size)
        field_vector = self.statistical_features_cached[0][:, :, 0]
        # (batch_size, graph_size, 1)
        return NodeFeatureExtractor.get_algebraic_distance(field_vector, sources)

    # @clock('AttentionModel')
    def get_graph_embed(self, embeds):
        """
        :param embeds: (batch_size, graph_size, embed_dim)
        :return:
        """
        if self.aggr_func == "sum":
            return embeds.sum(dim=1)
        elif self.aggr_func == "mean":
            return embeds.mean(dim=1)
        else:
            raise NotImplementedError(f"{self.aggr_func} has not been implemented")

    # @clock('AttentionModel')
    def _get_query(self, embeds, graph_embeds, first_nodes, graph_finished, step_counter):
        """
        :param embeds: (batch_size, graph_size, embed_dim)
        :param graph_embeds: (batch_size, embed_dim)
        :param first_nodes: List
        :param graph_finished: List
        :param step_counter: The number of env.step() invoked
        :return:
        """

        if step_counter % 2 == 0:  # Construct the context for choosing the first node.
            context_embed = torch.cat([graph_embeds,
                                       self.W_placeholder[None, :].expand(embeds.size(0), -1)  # (batch_size, embed_dim)
                                       ], dim=-1)  # (batch_size, 2 * embed_dim)
        else:  # Construct the context for choosing the second node.

            # first_nodes may contain None.
            first_nodes = np.array(first_nodes)  # List ----> Object type array
            # IMPORTANT : Do not change graph_finished to Torch.BoolTensor.
            first_nodes[graph_finished] = 0  # Replace None by 0
            first_nodes = first_nodes.astype('int64')  # Object type array ----> Int64 type array

            # (batch_size, ) --> (batch_size, 1, 1) --> (batch_size, 1, embed_dim)
            idx = torch.LongTensor(first_nodes)[:, None, None].expand(-1, -1, self.embed_dim).to(self.device)

            context_embed = torch.cat([graph_embeds,
                                       torch.gather(embeds, 1, idx).squeeze(1)],  # (batch_size, embed_dim)
                                      dim=-1)  # (batch_size, 2 * embed_dim)

        # (batch_size, 2 * embed_dim) --> (batch_size, embed_dim)
        return self.project_step_context(context_embed)

    def _make_heads(self, v):
        """
        :param v: (batch_size, graph_size, embed_dim)
        :return:
        """
        batch_size, graph_size, embed_dim = v.shape
        head_dim = embed_dim // self.n_heads
        # (batch_size, graph_size, embed_dim)
        # --> (batch_size, graph_size, n_heads, head_dim)
        # --> (n_heads, batch_size, graph_size, head_dim)
        return v.view(batch_size, graph_size, self.n_heads, head_dim).permute(2, 0, 1, 3)

    # @clock('AttentionModel')
    def _calculate_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """
        :param query: (batch_size, embed_dim)
        :param glimpse_K: (batch_size, graph_size, embed_dim)
        :param glimpse_V: (batch_size, graph_size, embed_dim)
        :param logit_K: (batch_size, graph_size, embed_dim)
        :param mask: (batch_size, graph_size)
        :return:
        """
        query = query.unsqueeze(1)  # (batch_size, embed_dim) --> (batch_size, 1, embed_dim)

        # query : (batch_size, 1, embed_dim) --> (n_heads, batch_size, 1, key_dim)
        # glimpse_K/glimpse_V : (batch_size, graph_size, embed_dim) --> (n_heads, batch_size, graph_size, key_dim)
        query, glimpse_K, glimpse_V = self._make_heads(query), self._make_heads(glimpse_K), self._make_heads(glimpse_V)

        # (n_heads, batch_size, 1, graph_size)
        compatibility = torch.matmul(query, glimpse_K.transpose(2, 3)) / sqrt(query.size(-1))

        compatibility = compatibility.masked_fill(mask[None, :, None, :].expand_as(compatibility), -math.inf)
        # compatibility[mask[None, :, None, :].expand_as(compatibility)] = -math.inf

        compatibility = torch.softmax(compatibility, dim=-1)

        # (n_heads, batch_size, 1, val_dim)
        # --> (n_heads, batch_size, val_dim)
        # --> (batch_size, n_heads, val_dim)
        # --> (batch_size, n_heads * val_dim)
        glimpse = torch.matmul(compatibility, glimpse_V) \
            .squeeze(2) \
            .permute(1, 0, 2) \
            .contiguous(). \
            view(glimpse_V.size(1), -1)

        # (batch_size, n_heads * val_dim) --> (batch_size, embed_dim)
        glimpse = self.project_out(glimpse)

        # (batch_size, 1, graph_size)
        logits = torch.matmul(glimpse[:, None, :], logit_K.transpose(1, 2)) / math.sqrt(glimpse.size(-1))
        logits = logits.squeeze(1)  # (batch_size, 1, graph_size) --> (batch_size, graph_size)

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        logits = logits.masked_fill(mask, -math.inf)
        # logits[mask] = -math.inf

        return logits  # (batch_size, graph_size)

    def set_decode_type(self, decode_type: str):
        self.decode_type = decode_type

    def _select_node(self, probs: torch.Tensor):
        """
        :param probs: (batch_size, graph_size)
        :return:
        """
        assert self.decode_type == "greedy" or self.decode_type == "sampling"

        selected = None
        if self.decode_type == "greedy":
            _, selected = probs.max(-1)  # (batch_size, )
        elif self.decode_type == "sampling":
            # (batch_size, graph_size) ----> (batch_size, 1) ----> (batch_size, )
            selected = probs.multinomial(1).squeeze(-1)

        return selected

    # @clock('AttentionModel')
    def _compute_log_p(self, env: NetworkEnhancementEnv):
        """
        :param env:
        :return: log_p : (batch_size, graph_size)
                 graph_finished : (batch_size, )
        """
        graphs, first_nodes, node_banned, graph_finished = env.get_state()

        # The graph structure changes only at 0, 2, 4, 6 ... steps.
        # Construct Statistical Features
        # Default Relative Features are 0.
        if env.step_counter % 2 == 0:
            statistical_features = [self.stat_feature_extractors[0](env.graphstates, graphs, self.node_dim),
                                    self.stat_feature_extractors[1](graphs),
                                    self.stat_feature_extractors[2](env.graphstates),
                                    self.stat_feature_extractors[3](graphs)
                                    ]

            # (batch_size, graph_size, self.relative_feature_dim)
            relative_features = [torch.zeros(statistical_features[0].size(0),
                                             statistical_features[0].size(1),
                                             self.relative_feature_dim)]
            self.statistical_features_cached = statistical_features
        else:
            statistical_features = self.statistical_features_cached
            relative_features = [extractor(graphs, first_nodes) for extractor in self.relative_feature_extractors]

        # (batch_size, graph_size, self.stat_feature_dim + self.relative_feature_dim)
        if not self.to_double:
            init_embed_ = torch.cat(statistical_features + relative_features, dim=-1).to(self.device)
        else:
            init_embed_ = torch.cat(statistical_features + relative_features, dim=-1).double().to(self.device)

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

        # Recover the mask.
        # mask = mask.masked_fill(torch.BoolTensor(graph_finished)[:, None].expand_as(mask), True)

        return log_p, graph_finished

    # @clock('AttentionModel')
    def act(self, env: NetworkEnhancementEnv):

        # log_p : (batch_size, graph_size)
        # graph_finished : (batch_size)
        log_p, graph_finished = self._compute_log_p(env)

        selected = self._select_node(log_p.exp())  # (batch_size, )

        selected[graph_finished] = -1  # To deal with the finished graph.

        return log_p, selected, graph_finished

    def evaluate(self, env: NetworkEnhancementEnv, actions):
        """
        :param env:
        :param actions: (batch_size, )
        :return:
        """
        log_p, graph_finished = self._compute_log_p(env)

        logpas_pred_batch = torch.gather(
            log_p,
            dim=-1,
            index=actions[:, None],
        ).squeeze(-1)  # (batch_size, )
        entropy_pred_batch = Categorical(probs=log_p.exp()).entropy()  # (batch_size, )
        return logpas_pred_batch, entropy_pred_batch

    # @clock('AttentionModel')
    def forward(self, env: NetworkEnhancementEnv):

        # otherwise, env will be modified in-place by env.step
        env = deepcopy(env)

        log_p_all_steps = []
        selected_all_steps = []
        graph_finished_all_steps = []

        while not env.is_terminal():
            log_p, selected, graph_finished = self.act(env)
            actions = [u if u != -1 else None for u in selected.tolist()]
            env.step(actions)

            log_p_all_steps.append(log_p)
            selected_all_steps.append(selected)
            graph_finished_all_steps.append(graph_finished)

        log_p_all_steps = torch.stack(log_p_all_steps, dim=1)  # (batch_size, step_num, graph_size)
        selected_all_steps = torch.stack(selected_all_steps, dim=1)  # (batch_size, step_num)
        graph_finished_all_steps = torch.stack(graph_finished_all_steps, dim=1)  # (batch_size, step_num)

        return log_p_all_steps, selected_all_steps, graph_finished_all_steps, env


if __name__ == '__main__':
    pass
    # Case 1
    # ---- Usage of func laplacian_matrix ----
    # g = nx.Graph()
    # elist = [(0, 1), (1, 2), (1, 3)]
    # g.add_edges_from(elist)
    #
    # L = nx.laplacian_matrix(g, nodelist=[0, 1, 2, 3])
    # L = L.astype(np.float32)
    #
    # L = np.array(L.todense())

    # ---- Usage of Numpy's linalg.eigh ----
    # import numpy.linalg as linalg
    # w, v = linalg.eigh(L)
    # print(w)
    # print(v)

    # ---- Stability of Numpy's linalg.eigh ----
    # w, v = linalg.eigh(L)
    # print(w)
    # print(v)

    # Case 2
    # ---- Euclidean Distance between nodes ----
    # The distance between adjacent nodes should be smaller than non-adjacent nodes.
    # g1 = nx.Graph()
    # elist1 = [(0, 1), (0, 2), (0, 3)]
    # g1.add_edges_from(elist1)
    # node_dim_ = 2
    # embed_dim_ = 16
    # attn = AttentionModel(node_dim_, embed_dim_)
    # node_embed = attn._get_shallow_embed([g1])
    # print("------****------")
    # print(node_embed)
    # print("------****------")
    # for i in range(g1.number_of_nodes()):
    #     for j in range(i + 1, g1.number_of_nodes()):
    #         print('Distance between node {0:d} and node {1:d} is {2:.2f}'.format(i, j, linalg.norm(node_embed[0, i]
    #                                                                                                - node_embed[0, j])
    #                                                                              )
    #               )

    # Case 3
    # ---- Test Forward Func of Attention Model  ----
    # g1 = nx.Graph()
    # g2 = nx.Graph()
    # g3 = nx.Graph()
    # elist1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # elist2 = [(0, 1), (0, 2), (0, 3)]
    # elist3 = [(0, 1), (0, 3), (2, 3)]
    # g1.add_edges_from(elist1)
    # g2.add_edges_from(elist2)
    # g3.add_edges_from(elist3)
    # node_dim_ = 2
    # embed_dim_ = 16
    #
    # graph_list = [nx.Graph(elist) for elist in [elist1, elist2, elist3]]
    # # env_ = NetworkEnhancementEnv([elist1, elist2, elist3], 20)
    # env_ = NetworkEnhancementEnv(graph_list, 20)
    # torch.manual_seed(1234)
    # attn = AttentionModel(node_dim_, embed_dim_)
    # attn.set_decode_type('greedy')
    # log_p_all_steps_, mask_all_steps_, selected_all_steps_, graph_finished_all_steps_ = attn(env_)
    # print("------ log_p_all_steps_ ------")
    # print(log_p_all_steps_)
    # print("------ mask_all_steps_ ------")
    # print(mask_all_steps_)
    # print("------ selected_all_steps_ ------")
    # print(selected_all_steps_)
    # print("------ graph_finished_all_steps_ ------")
    # print(graph_finished_all_steps_)

    # Case 4
    # ---- Test degree centrality  ----
    # g1 = nx.Graph()
    # g2 = nx.Graph()
    # g3 = nx.Graph()
    # elist1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # elist2 = [(0, 1), (0, 2), (0, 3)]
    # elist3 = [(0, 1), (0, 3), (2, 3)]
    #
    # graph_list = [nx.Graph(elist) for elist in [elist1, elist2, elist3]]
    #
    # print(AttentionModel.get_degree_centrality(graph_list))
