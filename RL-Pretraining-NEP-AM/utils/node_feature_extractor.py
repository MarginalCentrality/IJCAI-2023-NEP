import torch
import networkx as nx
import numpy as np
import numpy.linalg as linalg
from utils.functions import clock
import time


class NodeFeatureExtractor:

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_shallow_embed(graphstates, graphs, node_dim):
        """
        Get the shallow embeddings of nodes by
        :param graphstates:
        :param graphs:
        :param node_dim:
        :return:
        """

        if not graphstates[0].laplacian_available:
            nodes = list(range(graphs[0].number_of_nodes()))
            for idx, graph in enumerate(graphs):
                # csr_matrix with dtype=float32 ----> ndarray
                # (node_num, node_num)
                graphstates[idx].laplacian_available = True
                graphstates[idx].laplacian = np.array(nx.laplacian_matrix(graph,
                                                                          nodelist=nodes).astype(np.float32).todense())

        laplacians = [graphstate.laplacian for graphstate in graphstates]

        # (batch_size, graph_size, graph_size)
        laplacians = np.stack(laplacians, axis=0)

        # (batch_size, graph_size, graph_size)
        _, eigenvectors = linalg.eigh(laplacians)

        return torch.from_numpy(eigenvectors[:, :, 1: node_dim + 1])

    @staticmethod
    # Approximate
    def get_eigen_system(graphs):
        """
        Get the shallow embeddings of nodes
        :param graphs:
        :return:
        """

        laplacians = [None] * len(graphs)
        nodes = list(range(graphs[0].number_of_nodes()))

        for idx, graph in enumerate(graphs):
            # csr_matrix with dtype=float32 ----> ndarray
            # (node_num, node_num)
            laplacians[idx] = np.array(nx.laplacian_matrix(graph, nodelist=nodes).astype(np.float32).todense())

        # (batch_size, graph_size, graph_size)
        laplacians = np.stack(laplacians, axis=0)
        # eigenvalues : (batch_size, graph_size)
        # eigenvectors : (batch_size, graph_size, graph_size)
        eigenvalues, eigenvectors = linalg.eigh(laplacians)

        return eigenvalues, eigenvectors

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def approx_eigen(eigenvectors, eigenvalues, a, b, k, prev_res):
        """
        :param eigenvectors: (batch_size, graph_size, graph_size). eigenvectors[i, :, j] corresponds to the
        j-th eigenvector of the i-th graph.
        :param eigenvalues: (batch_size, graph_size)
        :param a: (batch_size, )
        :param b: (batch_size, )
        :param k: (node_dim, ), Eigenvectors to approximate.
        :param prev_res: The previous approximated eigenvectors, (batch_size, graph_size, node_dim)
        :return:
        """
        assert np.all(a < b)

        node_dim = k.shape[0]

        # (batch_size, 1, graph_size)
        numerator = (np.take_along_axis(eigenvectors, a[:, None, None], axis=1) -
                     np.take_along_axis(eigenvectors, b[:, None, None], axis=1))

        # (node_dim, batch_size, 1, graph_size)
        numerator = np.broadcast_to(numerator, (node_dim,) + numerator.shape)

        # (node_dim, ) ---> (node_dim, 1, ,1, 1)
        k = k[:, None, None, None]

        # np.take_along_axis(numerator, k, axis=3) : (node_dim, batch_size, 1, 1)
        # (node_dim, batch_size, 1, graph_size)
        numerator = numerator * np.take_along_axis(numerator, k, axis=3)

        # (node_dim, 1, 1, 1) ---> (node_dim, 1, 1)
        k = np.squeeze(k, -1)
        # (batch_size, graph_size) ---> (node_dim, batch_size, graph_size)
        eigenvalues = np.broadcast_to(eigenvalues, (node_dim,) + eigenvalues.shape)
        # (node_dim, batch_size, 1)
        denominator = np.take_along_axis(eigenvalues, k, axis=2)
        # (node_dim, batch_size, graph_size)
        denominator = denominator - eigenvalues
        # (node_dim, batch_size, 1, graph_size)
        denominator = denominator[:, :, None, :]

        # (node_dim, batch_size, 1, graph_size)
        idx = np.abs(denominator) < 1e-6
        numerator[idx] = 0.0
        denominator[idx] = 1.0

        # (node_dim, batch_size, graph_size) ---> (batch_size, graph_size, node_dim)
        return prev_res + np.transpose(
            np.matmul(np.broadcast_to(eigenvectors, (node_dim,) + eigenvectors.shape),
                      np.transpose(numerator / denominator, (0, 1, 3, 2))).squeeze(3),
            (1, 2, 0)
        )

    @staticmethod
    def approx_shallow_embed(graphstates, graphs, node_dim):
        """
        :param graphstates: List of graphstate, to approximate the shallow embeddings
        :param graphs: List of networkx graph
        :param node_dim:
        :return:
        """
        # No previous position encoding available.
        # We need last_inserted_edge because when performing beam search,
        # when choosing the first node, we compute the position encoding,
        # when choosing the second node, we would invoke this function again.
        # However, the position encoding is already available. Without last_inserted_edge,
        # we would meet a mistake.
        if not graphstates[0].approx_pos_enc_available:
            # eigenvalues : (batch_size, graph_size)
            # eigenvectors : (batch_size, graph_size, graph_size)
            eigenvalues, eigenvectors = NodeFeatureExtractor.get_eigen_system(graphs)

            # (batch_size, graph_size, node_dim)
            shallow_embed = eigenvectors[:, :, 1:node_dim + 1]

            # Update the graph state.
            for idx, graphstate in enumerate(graphstates):
                graphstate.eigenvalues = eigenvalues[idx]
                graphstate.eigenvectors = eigenvectors[idx]
                graphstate.approx_pos_enc = shallow_embed[idx]
                graphstate.approx_pos_enc_available = True
                graphstate.k = np.array(list(range(1, node_dim + 1)), dtype=np.int)

        else:
            shallow_embed = [graphstate.approx_pos_enc for graphstate in graphstates]
            # (batch_size, graph_size, node_dim)
            shallow_embed = np.stack(shallow_embed, axis=0)

        # Normalize the shallow_embed
        # (batch_size, 1, node_dim)
        normalization_factor = np.linalg.norm(shallow_embed,
                                              axis=1,
                                              keepdims=True)

        # (batch_size, graph_size, node_dim)
        shallow_embed = shallow_embed / normalization_factor
        return torch.from_numpy(shallow_embed)

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_fixed_shallow_embed(graphstates, graphs, node_dim):
        """
        Get the shallow embeddings of nodes by
        :param graphstates:
        :param graphs:
        :param node_dim:
        :return:
        """

        if not graphstates[0].fixed_pos_enc_available:
            nodes = list(range(graphs[0].number_of_nodes()))
            for idx, graph in enumerate(graphs):
                # csr_matrix with dtype=float32 ----> ndarray
                # (node_num, node_num)
                laplacian = np.array(nx.laplacian_matrix(graph,
                                                         nodelist=nodes).astype(np.float32).todense())

                _, eigenvectors = linalg.eigh(laplacian)

                graphstates[idx].fixed_pos_enc = eigenvectors[:, 1: node_dim + 1]
                graphstates[idx].fixed_pos_enc_available = True

        pos_enc = [graphstate.fixed_pos_enc for graphstate in graphstates]

        return torch.from_numpy(np.stack(pos_enc, axis=0))





    @staticmethod
    def _get_node_feature(graphs, extractor, params=None, default_val=None):
        """
        :param graphs: List of Networkx Graph
                       All graphs should have same node set.
        :param extractor: An Extractor Function
        :param params
        :param default_val
        :return:
        """
        node_features = []
        assert len(graphs) > 0
        nodes = list(range(graphs[0].number_of_nodes()))
        for idx, graph in enumerate(graphs):
            if params is None:
                node_feature = extractor(graph)  # map: node --> node_feature
            else:
                node_feature = extractor(graph, params[idx])

            if default_val is None:
                node_feature = np.array([node_feature[u] for u in nodes], dtype=np.float32)
            else:
                node_feature = np.array([node_feature.get(u, default_val) for u in nodes], dtype=np.float32)

            node_features.append(node_feature)
        # (batch_size, graph_size, 1)
        return torch.from_numpy(np.stack(node_features, axis=0)).unsqueeze(-1)

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_degree_centrality(graphs):
        """
        :param graphs: List of Networkx Graph
                       All graphs should have same node set.
        :return: (batch_size, graph_size， 1)
        """
        return NodeFeatureExtractor._get_node_feature(graphs, nx.degree_centrality)

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_core_number(graphs):
        """
        :param graphs: List of Networkx Graph
                       All graphs should have same node set.
        :return: (batch_size, graph_size, 1)
        """
        assert len(graphs) > 0
        # (batch_size, graph_size, 1)
        core_numbers = NodeFeatureExtractor._get_node_feature(graphs, nx.core_number)
        # Normalize the core number by number of nodes
        return core_numbers / graphs[0].number_of_nodes()

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_closeness_centrality(graphs):
        """
        :param graphs: List of Networkx Graph
                       All graphs should have same node set.
        :return: (batch_size, graph_size, 1)
        """
        return NodeFeatureExtractor._get_node_feature(graphs, nx.closeness_centrality)

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_current_flow_closeness_centrality(graphs):
        """
        :param graphs: List of Networkx Graph
                       All graphs should have same node set.
        :return: (batch_size, graph_size, 1)
        """
        return NodeFeatureExtractor._get_node_feature(graphs, nx.current_flow_closeness_centrality)

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_harmonic_centrality(graphs):
        """
        :param graphs: List of Networkx Graph
                       All graphs should have same node set.
        :return: (batch_size, graph_size, 1)
        """
        assert len(graphs) > 0
        # (batch_size, graph_size, 1)
        harmonic_centrality = NodeFeatureExtractor._get_node_feature(graphs, nx.harmonic_centrality)
        return harmonic_centrality / graphs[0].number_of_nodes()

    # @staticmethod
    # # @clock('NodeFeatureExtractor')
    # def get_clustering(graphs):
    #     """
    #     :param graphs: List of Networkx Graph
    #                    All graphs should have same node set.
    #     :return: (batch_size, graph_size, 1)
    #     """
    #     return NodeFeatureExtractor._get_node_feature(graphs, nx.clustering)


    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_clustering(graphstates):
        """
        :param graphstates: List of GraphState.
        :return: (batch_size, graph_size, 1)
        """
        if not graphstates[0].clustering_coefficient_available:
            nodes = list(range(graphstates[0].graph.number_of_nodes()))
            for graphstate in graphstates:
                triangles = nx.triangles(graphstate.graph)
                clustering_coefficient = nx.clustering(graphstate.graph)

                graphstate.triangles = np.array([triangles[u] for u in nodes], dtype=np.float32)
                graphstate.clustering_coefficient = np.array([clustering_coefficient[u] for u in nodes],
                                                             dtype=np.float32)
                graphstate.clustering_coefficient_available = True

        res = [graphstate.clustering_coefficient for graphstate in graphstates]

        # (batch_size, graph_size, 1)
        return torch.from_numpy(np.stack(res, axis=0)).unsqueeze(-1)



    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_average_neighbor_degree(graphs):
        """
        :param graphs: List of Networkx Graph
                       All graphs should have same node set.
        :return: (batch_size, graph_size, 1)
        """
        assert len(graphs) > 0
        # (batch_size, graph_size, 1)
        avg_nbor_degree = NodeFeatureExtractor._get_node_feature(graphs,
                                                                 nx.average_neighbor_degree)
        return avg_nbor_degree / graphs[0].number_of_nodes()

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_distance(graphs, sources):
        """
        :param graphs: List of Networkx Graph
                       All graphs should have same node set.
        :param sources:
        :return: (batch_size, graph_size, 1)
        """
        assert len(graphs) > 0
        nodes = list(range(graphs[0].number_of_nodes()))

        dist = []
        for src, graph in zip(sources, graphs):
            dist_map = nx.shortest_path_length(graph, source=src)
            dist.append(np.array([dist_map[u] for u in nodes], dtype=np.float32))

        # (batch_size, graph_size, 1)
        return torch.from_numpy(np.stack(dist, axis=0)).unsqueeze(-1) / graphs[0].number_of_nodes()

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_degree_product(graphs, sources):
        """
        :param graphs: List of Networkx Graph
                       All graphs should have same node set.
        :param sources:
        :return: (batch_size, graph_size)
        """
        # (batch_size, graph_size, 1)
        degree_centrality = NodeFeatureExtractor.get_degree_centrality(graphs)
        # (batch_size, )
        sources = torch.tensor(sources, dtype=torch.long)
        # (batch_size, 1, 1)
        src_deg_cent = torch.gather(degree_centrality, 1, sources[:, None, None])
        # (batch_size, graph_size, 1)
        return degree_centrality * src_deg_cent

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_algebraic_distance(field_vector, sources):
        """
        :param field_vector: (batch_size, graph_size)
        :param sources: List of Nodes
        :return:
        """
        # (batch_size, )
        sources = torch.tensor(sources, dtype=torch.long)
        # (batch_size, 1)
        src_field_val = torch.gather(field_vector, 1, sources[:, None])
        # (batch_size, graph_size, 1)
        # Normalized by 2.0
        return (torch.abs(field_vector - src_field_val) / 2.0).unsqueeze(-1)

    @staticmethod
    def _get_jaccard_coefficient(graph: nx, u):
        """
        :param graph: networkx obj
        :param u:
        :return: { w: jaccard_coefficient(u, w) }
                 Note :  If jaccard_coefficient(u, w) == 0, then w not in this dict.
        """
        node2jaccard = {}  # Map a node to its jaccard coefficient with u

        for v in graph.neighbors(u):
            for w in graph.neighbors(v):
                # if w != u:
                #     node2jaccard[w] = node2jaccard.setdefault(w, 0) + 1
                node2jaccard[w] = node2jaccard.setdefault(w, 0) + 1

        for w in node2jaccard:
            node2jaccard[w] = node2jaccard[w] / (graph.degree(u) + graph.degree(w) - node2jaccard[w])

        return node2jaccard

    @staticmethod
    # @clock('NodeFeatureExtractor')
    def get_jaccard_coefficient(graphs, sources):
        """
        :param graphs: List of Networkx Graph
                       All graphs should have same node set.
        :param sources:
        :return:
        """
        # (batch_size, graph_size, 1)
        return NodeFeatureExtractor._get_node_feature(graphs,
                                                      NodeFeatureExtractor._get_jaccard_coefficient,
                                                      params=sources,
                                                      default_val=0.0)


if __name__ == '__main__':

    edge_lists = [[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [(0, 1), (0, 3), (1, 2), (2, 3)],
                  [(0, 1), (0, 2), (1, 3)]]

    graphs_ = [nx.Graph() for _ in range(len(edge_lists))]
    for i, edge_list in enumerate(edge_lists):
        graphs_[i].add_edges_from(edge_list)

    print(NodeFeatureExtractor.get_degree_centrality(graphs_))
    print(NodeFeatureExtractor.get_core_number(graphs_))
    print(NodeFeatureExtractor.get_closeness_centrality(graphs_))
    print(NodeFeatureExtractor.get_current_flow_closeness_centrality(graphs_))
    print(NodeFeatureExtractor.get_harmonic_centrality(graphs_))
    print(NodeFeatureExtractor.get_clustering(graphs_))
    print(NodeFeatureExtractor.get_average_neighbor_degree(graphs_))

    sources_ = [0, 1, 1]
    print(NodeFeatureExtractor.get_distance(graphs_, sources_))
    print(NodeFeatureExtractor.get_degree_product(graphs_, sources_))

    # (batch_size, graph_size, node_dim)
    shallow_embed = NodeFeatureExtractor.get_shallow_embed(graphs_, node_dim=1)
    # (batch_size, graph_size)
    field_vector = shallow_embed.squeeze(dim=-1)

    print(NodeFeatureExtractor.get_algebraic_distance(field_vector, sources_))
    print(NodeFeatureExtractor.get_jaccard_coefficient(graphs_, sources_))

    # from itertools import product

    # for idx, src in enumerate(sources_):
    #     nodes = list(range(graphs_[0].number_of_nodes()))
    #     ebunch = product([src], nodes)
    #     for u, v, p in nx.jaccard_coefficient(graphs_[idx], ebunch):
    #         print(u, v, p)
