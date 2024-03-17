import torch
from torch import nn
import numpy as np
from nets.graph_encoder import GraphAttentionEncoder
from nets.attention_model import AttentionModel
from problems.network_enhancement_env import NetworkEnhancementEnv


class CriticNetwork(nn.Module):
    def __init__(self,
                 node_dim,
                 embed_dim,
                 hidden_dim,
                 n_heads=8,
                 n_layers=2):
        super(CriticNetwork, self).__init__()

        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.init_embed = nn.Linear(self.node_dim, self.embed_dim)
        # The hidden dim of feed forward module is 512 by default.
        self.embedder = GraphAttentionEncoder(
            self.n_heads,
            self.embed_dim,
            self.n_layers
        )

        self.W_placeholder = nn.Parameter(torch.Tensor(embed_dim))
        self.W_placeholder.data.uniform_(-1, 1)

        # (batch_size, 2 * embed_dim) ----> (batch_size, hidden_dim)
        # (batch_size, hidden_dim) ----> (batch_size, 1)
        self.value_head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        # Cached Tensors
        self.embeds_cached = None
        self.graph_embeds_cached = None

    def _get_context(self, embeds, graph_embeds, first_nodes, graph_finished, step_counter):
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

            # TODO: Move the tensor to the right device.
            # (batch_size, ) --> (batch_size, 1, 1) --> (batch_size, 1, embed_dim)
            idx = torch.LongTensor(first_nodes)[:, None, None].expand(-1, -1, self.embed_dim)
            context_embed = torch.cat([graph_embeds,
                                       torch.gather(embeds, 1, idx).squeeze(1)],  # (batch_size, embed_dim)
                                      dim=-1)  # (batch_size, 2 * embed_dim)

        # (batch_size, 2 * embed_dim)
        return context_embed

    def forward(self, env: NetworkEnhancementEnv):
        graphs, first_nodes, node_banned, graph_finished = env.get_state()

        embeds = self.embeds_cached
        graph_embeds = self.graph_embeds_cached
        # The graph structure changes only at 0, 2, 4, 6 ... steps.
        if env.step_counter % 2 == 0:
            shallow_embed_ = AttentionModel.get_shallow_embed(graphs,
                                                              self.node_dim)  # (batch_size, graph_size, node_dim)
            init_embed_ = self.init_embed(shallow_embed_)  # (batch_size, graph_size, embed_dim)
            embeds = self.embedder(init_embed_)  # (batch_size, graph_size, embed_dim)
            graph_embeds = AttentionModel.get_graph_embed(embeds)  # (batch_size, embed_dim)

            # Update Cache
            self.embeds_cached = embeds
            self.graph_embeds_cached = graph_embeds

        # (batch_size, 2 * embed_dim)
        context_embed = self._get_context(embeds, graph_embeds, first_nodes, graph_finished, env.step_counter)

        # (batch_size, 2 * embed_dim) ----> (batch_size, )
        return self.value_head(context_embed).squeeze(-1)


if __name__ == '__main__':
    import networkx as nx
    # ---- Test Forward Func of CriticNetwork  ----
    g1 = nx.Graph()
    g2 = nx.Graph()
    g3 = nx.Graph()
    elist1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    elist2 = [(0, 1), (0, 2), (0, 3)]
    elist3 = [(0, 1), (0, 3), (2, 3)]
    g1.add_edges_from(elist1)
    g2.add_edges_from(elist2)
    g3.add_edges_from(elist3)
    node_dim_ = 2
    embed_dim_ = 16
    hidden_dim_ = 32
    # env_ = NetworkEnhancementEnv([elist1, elist2, elist3], 20)
    env_ = NetworkEnhancementEnv([g1, g2, g3], 20)
    torch.manual_seed(1234)
    critic = CriticNetwork(node_dim_, embed_dim_, hidden_dim_)
    print(critic(env_))
