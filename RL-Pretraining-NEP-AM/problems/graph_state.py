import time

import networkx as nx
from math import ceil
import numpy as np

from utils.node_feature_extractor import NodeFeatureExtractor


class GraphState:

    def __init__(self, graph: nx.Graph, edge_budget_percentage, seed=None):
        # def __init__(self, edge_list, edge_budget_percentage, name):
        """
        :param graph: a networkx object
        :param edge_budget_percentage
        """
        # self.graph = nx.Graph(edge_list) if name is None else nx.Graph(edge_list, name=name)

        self.graph = graph
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))  # Do not consider self-loops
        self.edge_budget_percentage = edge_budget_percentage
        self.edge_budget = self._get_edge_budget()
        self.first_node = None

        # To Deal with the rollout baseline
        self.seed = seed

        # To Deal with the fast shallow embedding
        self.laplacian_available = False
        self.laplacian = None

        # To Deal with the approximated position encoding
        self.approx_pos_enc_available = False
        self.approx_pos_enc = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.k = None

        # To Deal with the fixed position encoding
        self.fixed_pos_enc_available = False  # the availability of position encoding
        self.fixed_pos_enc = None  # value of position encoding

        # To Deal with the clustering coefficient
        self.clustering_coefficient_available = False
        self.triangles = None
        self.clustering_coefficient = None

        self.node_banned = np.full(self.graph.number_of_nodes(), False, dtype=bool)
        self.node_banned[self._get_invalid_first_node()] = True
        # self.banned_nodes = self._get_invalid_first_node()

        self.finished = np.all(self.node_banned)
        # self.finished = False if len(self.banned_nodes) < self.graph.number_of_nodes() else True


    def _get_edge_budget(self):
        num_node = self.graph.number_of_nodes()
        edge_budget = ceil((num_node * (num_node - 1)) / 2.0 * self.edge_budget_percentage / 100.0)
        assert edge_budget > 0, 'Error : No extra budget at initializing.'
        return edge_budget

    # Get nodes that cannot be chosen as first node.
    def _get_invalid_first_node(self):
        graph_size = self.graph.number_of_nodes()
        invalid_node_list = []
        for u, deg in self.graph.degree:
            if deg == graph_size - 1:
                invalid_node_list.append(u)
        return invalid_node_list

    # Get nodes that already have a connection with u.
    def _get_invalid_edge_ends(self, u):
        assert u in self.graph, 'Error : Node {0:d} is not in graph {1:d}.'.format(u, self.graph.name)
        return list(self.graph.neighbors(u)) + [u]  # Do not consider self-loop.

    def _adjust_clustering_coefficient(self, u, v):
        """
        u, v are nodes where an edge is inserted.
        """
        common_nbors = list(nx.common_neighbors(self.graph, u, v))
        n = len(common_nbors)

        self.triangles[u] = self.triangles[u] + n
        self.triangles[v] = self.triangles[v] + n

        for w in common_nbors:
            self.triangles[w] = self.triangles[w] + 1

        for w in common_nbors + [u, v]:
            self.clustering_coefficient[w] = (2 * self.triangles[w]) / (self.graph.degree(w) * (self.graph.degree(w) - 1))

    # in-place update
    def update(self, u):
        if not self.finished:
            assert u in self.graph, 'Error : Node {0:d} is not in graph {1:s}.'.format(u, self.graph.name)
            assert not self.node_banned[u], 'Error : Node {0:d} ' \
                                            'has been banned in graph {1:s}'.format(u, self.graph.name)
            # assert u not in self.banned_nodes, 'Error : Node {0:d} ' \
            #                                    'has been banned in graph {1:d}'.format(u, self.graph.name)

            self.node_banned[:] = False
            if self.first_node is None:
                self.first_node = u
                self.node_banned[self._get_invalid_edge_ends(self.first_node)] = True
                # self.banned_nodes = self._get_invalid_edge_ends(self.first_node)
            else:
                self.graph.add_edge(self.first_node, u)

                if self.laplacian_available:
                    self.laplacian[self.first_node, self.first_node] = self.laplacian[self.first_node,
                                                                                      self.first_node] + 1
                    self.laplacian[u, u] = self.laplacian[u, u] + 1

                    self.laplacian[self.first_node, u] = self.laplacian[self.first_node, u] - 1
                    self.laplacian[u, self.first_node] = self.laplacian[u, self.first_node] - 1

                if self.clustering_coefficient_available:
                    self._adjust_clustering_coefficient(self.first_node, u)

                if self.approx_pos_enc_available:
                    # (batch_size, graph_size, graph_size)
                    eigenvectors = np.expand_dims(self.eigenvectors, axis=0)
                    # (batch_size, graph_size)
                    eigenvalues = np.expand_dims(self.eigenvalues, axis=0)

                    # (batch_size, )
                    last_inserted_edge = [self.first_node, u] if self.first_node < u \
                        else [u, self.first_node]

                    a = np.array([last_inserted_edge[0]])
                    b = np.array([last_inserted_edge[1]])
                    # (batch_size, graph_size, node_dim)
                    prev_res = np.expand_dims(self.approx_pos_enc, axis=0)

                    # (graph_size, node_dim)
                    self.approx_pos_enc = NodeFeatureExtractor.approx_eigen(eigenvectors,
                                                                            eigenvalues,
                                                                            a,
                                                                            b,
                                                                            self.k,
                                                                            prev_res).squeeze(0)

                self.edge_budget = self.edge_budget - 1
                self.first_node = None
                banned_nodes = list(self.graph.nodes) if self.edge_budget == 0 else self._get_invalid_first_node()
                self.node_banned[banned_nodes] = True
                self.finished = np.all(self.node_banned)

        else:
            print('Warning : No updates in finished graph {0:d}.'.format(self.graph.name))

    def __str__(self):
        nodes = ' '.join(str(u) for u in self.graph.nodes)
        edges = '\n'.join(map(lambda x: '---'.join([str(x[0]), str(x[1])]), self.graph.edges))
        banned_nodes = ' '.join(str(u) for u in np.where(self.node_banned == 1)[0])
        return 'name : {0:d}\n'.format(self.graph.name) + \
               'nodes : ' + nodes + '\n' \
               + 'edges : \n' + edges + '\n' \
               + 'edge budget : ' + str(self.edge_budget) + '\n' \
               + 'first node : ' + str(self.first_node) + '\n' \
               + 'banned nodes : ' + banned_nodes + '\n' \
               + 'finished state : ' + str(self.finished) + '\n'


if __name__ == "__main__":
    # edge_list_ = [(0, 1), (0, 2)]
    # graph_ = nx.Graph(edge_list_, name=0)
    # edge_budget_percentage_ = 20
    # a = GraphState(graph_, edge_budget_percentage_)
    # print(a)
    # a.update(2)
    # print(a)
    # a.update(1)
    # print(a)

    # edge_list_ = [(0, 1), (0, 2), (0, 3), (1, 2)]
    # graph_ = nx.Graph(edge_list_, name=0)
    # graph_ = nx.barabasi_albert_graph(n=100, m=2)

    graph_ = nx.gnp_random_graph(n=100, p=0.3)

    edge_budget_percentage_ = 100
    a = GraphState(graph_, edge_budget_percentage_)

    a.clustering_coefficient_available = True
    a.triangles = nx.triangles(a.graph)

    nodes = list(a.graph.nodes)

    a.clustering_coefficient = nx.clustering(a.graph)

    n_edge = 1000
    from random import choices
    elapsed = {'clustering': 0.0, 'adjust': 0.0}

    for i in range(n_edge):

        while True:
            u, v = choices(nodes, k=2)
            if u != v and not a.graph.has_edge(u, v):
                break

        # print((u, v))
        a.graph.add_edge(u, v)

        start_time = time.perf_counter()
        a._adjust_clustering_coefficient(u, v)
        end_time = time.perf_counter()
        elapsed['adjust'] = elapsed['adjust'] + (end_time - start_time)

        start_time = time.perf_counter()
        res = nx.clustering(a.graph)
        end_time = time.perf_counter()
        elapsed['clustering'] = elapsed['clustering'] + (end_time - start_time)

        # for node, coeff in a.clustering_coefficient.items():
        #     if coeff != res[node]:
        #         print(node, coeff, res[node])

        print(a.clustering_coefficient == nx.clustering(a.graph))
    print(elapsed['adjust'])
    print(elapsed['clustering'])