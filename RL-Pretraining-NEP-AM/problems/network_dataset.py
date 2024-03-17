from torch.utils.data import Dataset
from problems.network_generator import BANetworkGenerator, GNMNetworkGenerator
from utils.functions import list2dict
import networkx as nx


class NetworkDataset(Dataset):
    def __init__(self, random_seeds, opts):
        """
        :param random_seeds:
        :param opts:
               -- graph_model
               -- store_graphs
               -- graph_storage_root
               -- delta_seed
        """

        self.random_seeds = random_seeds

        # Generate or Load data from disk
        # If the graphs have been generated, they will be loaded from the disk directly.
        graph_model, *params = opts.graph_model.split('_')
        self.graph_model = graph_model
        self.params = list2dict(params)

        self.opts = opts
        self.data = self._get_dataset()

    def _get_dataset(self):
        if self.graph_model == 'BA':
            generator = BANetworkGenerator(self.opts.store_graphs,
                                           self.opts.graph_storage_root)
        elif self.graph_model == 'GNM':
            generator = GNMNetworkGenerator(self.opts.store_graphs,
                                            self.opts.graph_storage_root)
        else:
            raise NotImplementedError(f"{self.graph_model} has not been implemented")

        graphs = generator.generate_multi_instances(self.params,
                                                    self.random_seeds,
                                                    self.opts.delta_seed)

        # Convert nx object to numpy array for Dataloader to process.
        return [nx.to_numpy_array(graph) for graph in graphs]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)



class EnhancedNetworkDataset(NetworkDataset):
    def __init__(self, random_seeds, opts):
        """
        :param random_seeds:
        :param opts:
               -- graph_model
               -- store_graphs
               -- graph_storage_root
               -- delta_seed
        """

        super().__init__(random_seeds, opts)

    def __getitem__(self, idx):
        return self.data[idx], self.random_seeds[idx]



