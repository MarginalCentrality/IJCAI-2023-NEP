import os
from abc import abstractmethod

import networkx as nx


def get_data_filename(gen_params: dict, random_seed):
    n = gen_params['n']
    return f"{n}-{random_seed}.graphml"


def get_drawing_filename(gen_params, random_seed):
    n = gen_params['n']
    return f"{n}-{random_seed}.png"


def draw_to_file(filepath, instance: nx.Graph):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_size_length = instance.number_of_nodes() / 5
    fig_size = (fig_size_length, fig_size_length)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    nx.draw_shell(instance, with_labels=True, ax=ax)
    fig.savefig(filepath)
    plt.close()


def construct_network_seeds(n_train, n_val, n_test):
    train_seeds = list(range(n_train))
    val_seeds = list(range(n_train, n_train + n_val))
    test_seeds = list(range(n_train + n_val, n_train + n_val + n_test))
    return train_seeds, val_seeds, test_seeds



class NetworkGenerator:
    def __init__(self, name, store_graphs=False, graph_storage_root=None):
        super().__init__()
        self.name = name
        self.store_graphs = store_graphs
        self.graph_storage_root = graph_storage_root

        if self.graph_storage_root is not None:
            self.graph_storage_dir = os.path.join(self.graph_storage_root, self.name)

        if self.store_graphs:
            os.makedirs(self.graph_storage_dir, exist_ok=True)

    @abstractmethod
    def _generate_instance(self, gen_params, random_seed, delta_seed):
        pass

    def _generate(self, gen_params, random_seed, delta_seed):
        """
        :param gen_params:
        :param random_seed:
        :param delta_seed:
         - - If random_seed is fail, e.g., in generating a connected network,
         - - then random_seed + delta_seed will be used.
        :return:
        """
        filename = get_data_filename(gen_params, random_seed)

        is_graph_on_disk = False
        if self.graph_storage_root is not None:
            filepath = os.path.join(self.graph_storage_dir, filename)
            is_graph_on_disk = os.path.exists(filepath)

        # Read the graph instance from disk, if it exists.
        if is_graph_on_disk:
            # print('Read from Disk')
            instance = nx.read_graphml(filepath, node_type=int)
        else:
            # Generate the instance
            instance = self._generate_instance(gen_params, random_seed, delta_seed)
            # Store the instance into the disk.
            if self.store_graphs:
                nx.write_graphml(instance, filepath)  # Write the graph into disk.
                # Draw the graph
                drawing_filename = get_drawing_filename(gen_params, random_seed)
                drawing_path = os.path.join(self.graph_storage_dir, drawing_filename)
                draw_to_file(drawing_path, instance)

        return instance

    def generate_multi_instances(self, gen_params, random_seeds, delta_seed):
        return [self._generate(gen_params, random_seed, delta_seed) for random_seed in random_seeds]


class GNMNetworkGenerator(NetworkGenerator):

    def __init__(self, store_graphs=False,
                 graph_storage_root=None,
                 enforce_connected=True):
        super().__init__('random_network', store_graphs, graph_storage_root)
        self.num_tries = 10000
        self.enforce_connected = enforce_connected
        self.generator = nx.generators.random_graphs.gnm_random_graph

    def _generate_instance(self, gen_params, random_seed, delta_seed):
        """
        :param gen_params:
        :param random_seed:
        :param delta_seed:
        :return:
        """

        n = int(gen_params['n'])  # Number of nodes
        m = int(gen_params['m'])  # Number of edges

        instance = self.generator(n, m, random_seed)
        is_connected = nx.is_connected(instance)

        idx = 0  # idx of attempting to generate a connected graph
        while not is_connected and idx < self.num_tries:
            instance = self.generator(n, m, random_seed + delta_seed * idx)
            is_connected = nx.is_connected(instance)
            idx += 1
            print(f"Have tried {idx} times !!!")

        if is_connected:
            return instance
        else:
            raise ValueError("Maximum number of tries exceeded, giving up ...")


class BANetworkGenerator(NetworkGenerator):
    def __init__(self, store_graphs=False,
                 graph_storage_root=None):
        super().__init__('barabasi_albert', store_graphs, graph_storage_root)
        self.generator = nx.generators.random_graphs.barabasi_albert_graph

    def _generate_instance(self, gen_params, random_seed, delta_seed):
        n = int(gen_params['n'])
        m = int(gen_params['m'])  # Number of edges to attach from a new node to existing nodes
        return self.generator(n, m, random_seed)


if __name__ == '__main__':
    # Case 1 : Test GNMNetworkGenerator
    # gen_params_ = {'n': 30, 'm': 31}
    # gnm = GNMNetworkGenerator(store_graphs=True, graph_storage_root='./GeneratedGraphTest/')
    # random_seeds_ = list(range(3))
    # instances = gnm.generate_multi_instances(gen_params_, random_seeds_, len(random_seeds_))
    # for instance in instances:
    #     print(instance.nodes)

    # Case 2 : Test BANetworkGenerator
    gen_params_ = {'n': 30, 'm': 2}
    # BA = BANetworkGenerator(store_graphs=True, graph_storage_root='./GeneratedGraphTest/')
    BA = BANetworkGenerator(graph_storage_root='./GeneratedGraphTest/')
    random_seeds_ = list(range(3))
    instances = BA.generate_multi_instances(gen_params_, random_seeds_, len(random_seeds_))
    for instance_ in instances:
        print(instance_.nodes)
