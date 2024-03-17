from abc import abstractmethod
import numpy as np
import networkx as nx


class RobustnessEstimation:
    def __init__(self, graph_nx, num_mc_sims):
        self.graph_nx = graph_nx
        self.num_mc_sims = num_mc_sims

    @abstractmethod
    def get_targets(self):
        pass

    def get_robustness_estimation(self):
        estimations = np.zeros(self.num_mc_sims)
        num_node = self.graph_nx.number_of_nodes()
        for i in range(self.num_mc_sims):
            # The graph structure is deeply copied in below method.
            graph = self.graph_nx.copy()
            targets = self.get_targets()
            for j, target in enumerate(targets):
                graph.remove_node(target)
                if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
                    estimations[i] = (j + 1) / num_node
                    break

        return estimations.mean()








