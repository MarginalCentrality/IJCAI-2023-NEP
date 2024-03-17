import networkx as nx
from problems.graph_state import GraphState
from problems.random_removal import RandomRemoval
from problems.targeted_removal import TargetedRemoval
from utils.functions import clock
from mpire import WorkerPool


class NetworkEnhancementEnv:

    # def __init__(self, graph_list, edge_budget_percentage):
    #     """
    #     :param graph_list: Each graph is an object of networkx
    #     :param edge_budget_percentage: In Percentile.
    #     """
    #
    #     # self.graphstates = [GraphState(g, edge_budget_percentage, idx) for idx, g in enumerate(graph_list)]
    #     self.graphstates = [GraphState(graph, edge_budget_percentage) for graph in graph_list]
    #     self.batch_size = len(graph_list)
    #     self.step_counter = 0  # The number of function step() invoked

    def __init__(self, graphstates):
        """
        :param graphstates: A List of Graph States.
        """

        self.graphstates = graphstates
        self.batch_size = len(graphstates)
        self.step_counter = 0  # The number of function step() invoked

    def get_state(self):
        states = [(graph_state.graph, graph_state.first_node, graph_state.node_banned, graph_state.finished)
                  for graph_state in self.graphstates]
        return list(list(item) for item in zip(*states))

    # @clock('NetworkEnhancementEnv')
    def step(self, actions):
        """
        :param actions: an action can be a node or None.
        :return:
        """
        assert self.batch_size == len(actions), 'Error : Apply one action to each graph.'
        for u, graph_state in zip(actions, self.graphstates):
            if u is not None:
                graph_state.update(u)

        self.step_counter += 1

    # TODO : Write a multi-process version of this function.
    # @clock('NetworkEnhancementEnv')
    def calculate_robustness(self, method, num_mc_sims, seed=None, reuse_hash=False):
        assert method == RandomRemoval or method == TargetedRemoval, "method has not been implemented."
        if method == RandomRemoval:
            instances = [method(graph_state.graph, num_mc_sims, seed, reuse_hash)
                         for graph_state in self.graphstates]
        else:
            instances = [method(graph_state.graph, num_mc_sims) for graph_state in self.graphstates]

        return [instance.get_robustness_estimation() for instance in instances]

    def _calculate_robustness(self, instance):
        return instance.get_robustness_estimation()

    def calculate_robustness_parallel(self, method, num_mc_sims, n_process, seed=None, reuse_hash=False):
        # Set start_method='spawn' to avoid "Cannot re-initialize CUDA in forked subprocess."
        assert method == RandomRemoval or method == TargetedRemoval, "method has not been implemented."
        if method == RandomRemoval:
            instances = [method(graph_state.graph, num_mc_sims, seed, reuse_hash)
                         for graph_state in self.graphstates]
        else:
            instances = [method(graph_state.graph, num_mc_sims) for graph_state in self.graphstates]


        with WorkerPool(n_jobs=n_process, start_method='spawn') as pool:
            robustness = pool.map(self._calculate_robustness, instances)

        return robustness

    def is_terminal(self):
        flags = [graph_state.finished for graph_state in self.graphstates]
        return all(flags)


if __name__ == '__main__':
    # Case 1
    # ---- Test the interactions with ENV  ----
    edge_list_ = [
        [(0, 1), (1, 2)],
        [(0, 1), (0, 2)],
        [(0, 1)]
    ]
    graph_list_ = [nx.Graph(edges_) for edges_ in edge_list_]
    num_mc_sims = 100
    env = NetworkEnhancementEnv(graph_list_, 20)
    graphs, first_nodes, node_banned, graph_finished = env.get_state()

    print("---------------")
    [print(g) for g in graphs]
    print(first_nodes)
    print(node_banned)
    print(graph_finished)
    print("Robustness of Random Removal")
    print(env.calculate_robustness(RandomRemoval, num_mc_sims))
    print("Robustness of Targeted Removal")
    print(env.calculate_robustness(TargetedRemoval, num_mc_sims))

    actions_ = [0, 1, None]
    env.step(actions_)
    graphs, first_nodes, node_banned, graph_finished = env.get_state()

    print("---------------")
    [print(g) for g in graphs]
    print(first_nodes)
    print(node_banned)
    print(graph_finished)
    print("Robustness of Random Removal")
    print(env.calculate_robustness(RandomRemoval, num_mc_sims))
    print("Robustness of Targeted Removal")
    print(env.calculate_robustness(TargetedRemoval, num_mc_sims))

    actions_ = [2, 2, None]
    env.step(actions_)
    graphs, first_nodes, node_banned, graph_finished = env.get_state()

    print("---------------")
    [print(g) for g in graphs]
    print(first_nodes)
    print(node_banned)
    print(graph_finished)
    print("Robustness of Random Removal")
    print(env.calculate_robustness(RandomRemoval, num_mc_sims))
    print("Robustness of Targeted Removal")
    print(env.calculate_robustness(TargetedRemoval, num_mc_sims))
