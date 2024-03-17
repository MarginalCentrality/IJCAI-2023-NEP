import random as rd
import xxhash
from problems.robustness_estimation import RobustnessEstimation
import networkx as nx


class RandomRemoval(RobustnessEstimation):
    # Share hash between the Initial Network and the Modified Network,
    # such that the attack sequences are the same between them.
    hash_to_reuse = None

    def __init__(self, graph_nx, num_mc_sims, seed=None, reuse_hash=False):
        super(RandomRemoval, self).__init__(graph_nx, num_mc_sims)

        self.seed = seed
        self.reuse_hash = reuse_hash
        assert reuse_hash is False or seed is not None

        if reuse_hash:
            if RandomRemoval.hash_to_reuse is None:
                RandomRemoval.hash_to_reuse = self._get_graph_hash()

            self.graph_hash = RandomRemoval.hash_to_reuse
        else:
            self.graph_hash = self._get_graph_hash()
        self.nodes = list(self.graph_nx.nodes())

    def get_targets(self):
        rd.shuffle(self.nodes)
        return self.nodes

    def _get_graph_hash(self):
        hash_instance = xxhash.xxh32()
        hash_instance.update(nx.to_numpy_array(self.graph_nx))
        return hash_instance.intdigest()

    def get_robustness_estimation(self):
        if self.seed is None:
            return super(RandomRemoval, self).get_robustness_estimation()
        else:
            rd.seed(self.graph_hash * self.seed)
            res = super(RandomRemoval, self).get_robustness_estimation()

        return res


if __name__ == '__main__':
    import networkx as nx

    g = nx.Graph()
    elist = [(1, 2), (2, 3), (3, 4), (4, 1)]
    g.add_edges_from(elist)
    num_mc_sims_ = 10
    rand_removal = RandomRemoval(g, num_mc_sims_, seed=1234)
    # rand_removal = RandomRemoval(g, num_mc_sims_)
    # for i in range(num_mc_sims_):
    #     print(rand_removal.get_targets())
    print(rand_removal.get_robustness_estimation())
