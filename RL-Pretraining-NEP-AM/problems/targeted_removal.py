from problems.robustness_estimation import RobustnessEstimation


class TargetedRemoval(RobustnessEstimation):

    def __init__(self, graph_nx, num_mc_sims):
        if num_mc_sims > 1:  # No randomness in targeted removal by degree
            num_mc_sims = 1

        super(TargetedRemoval, self).__init__(graph_nx, num_mc_sims)

        self.targets = None

    def get_targets(self):
        if self.targets is None:
            self.targets = [item[0] for item in sorted(self.graph_nx.degree, key=lambda x: x[1], reverse=True)]

        return self.targets


if __name__ == '__main__':
    import networkx as nx

    g = nx.Graph()
    elist = [(1, 2), (2, 3), (2, 4)]
    g.add_edges_from(elist)
    num_mc_sims_ = 1000000
    targeted_removal = TargetedRemoval(g, num_mc_sims_)
    print(targeted_removal.get_robustness_estimation())
