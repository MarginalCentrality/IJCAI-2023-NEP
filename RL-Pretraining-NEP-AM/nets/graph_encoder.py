import torch
from torch import nn
from math import sqrt
from utils.functions import clock


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 n_heads,
                 input_dim,
                 embed_dim,
                 val_dim=None,
                 key_dim=None):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1.0 / sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.W_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.W_val = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.val_dim))

        self.W_out = nn.Parameter(torch.Tensor(self.n_heads * self.val_dim, self.embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h):
        """
        :param h: (batch_size, graph_size, input_dim)
        :return:
        """
        batch_size, graph_size, input_dim = h.size()
        assert self.input_dim == input_dim, "Wrong embedding dimension of input."

        # (batch_size * graph_size, input_dim)
        hflat = h.view(-1, self.input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        # Calculate queries and keys: (n_heads, batch_size, graph_size, key_dim)
        Q = torch.matmul(hflat, self.W_query).view(shp)
        K = torch.matmul(hflat, self.W_key).view(shp)

        # Calculate keys and values: (n_heads, batch_size, graph_size, val_dim)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility: (n_heads, batch_size, graph_size, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Calculate attention: (n_heads, batch_size, graph_size, graph_size)
        attn = torch.softmax(compatibility, -1)

        # Calculate heads: (n_heads, batch_size, graph_size, val_dim)
        heads = torch.matmul(attn, V)

        # (batch_size, graph_size, embed_dim)
        out = torch.matmul(
            heads.permute(1, 2, 0, 3).contiguous().view(batch_size, graph_size, -1),
            self.W_out
        )
        return out


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input_):
        return input_ + self.module(input_)


class Normalization(nn.Module):
    def __init__(self, embed_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)

    def forward(self, input_):
        """
        :param input_: (batch_size, graph_size, embed_dim)
        :return:
        """
        return self.normalizer(input_.view(-1, input_.size(-1))).view(input_.size())


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(self,
                 n_heads,
                 embed_dim,
                 feed_forward_hidden=512):
        super(MultiHeadAttentionLayer, self).__init__(
            # (batch_size, graph_size, embed_dim)
            SkipConnection(
                MultiHeadAttention(n_heads,
                                   embed_dim,
                                   embed_dim)
            ),
            Normalization(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(self,
                 n_heads,
                 embed_dim,
                 n_layers,
                 feed_forward_hidden=512):
        super(GraphAttentionEncoder, self).__init__()

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        )
                                    )

    #@clock(tag='GraphAttentionEncoder')
    def forward(self, h):
        """
        :param h: (batch_size, graph_size, input_dim)
        :return:
        """

        return self.layers(h)


if __name__ == '__main__':
    n_heads_ = 8
    input_dim_ = 256
    embed_dim_ = 256
    batch_size_ = 16
    graph_size_ = 20

    # mha = MultiHeadAttention(n_heads_, input_dim_, embed_dim_)
    h_ = torch.randn(batch_size_, graph_size_, input_dim_)

    # normalizer = Normalization(embed_dim_)

    # res = normalizer(mha(h_))
    #
    # print(res.size())
