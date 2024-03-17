import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from utils.node_feature_extractor import NodeFeatureExtractor

data_dir = '../data/barabasi_albert'
res_dir = data_dir
idx = list(range(0, 10))
graphs = [f"20-{i}.graphml" for i in idx]
params = [10] * 10
x_label = "jaccard_coefficient-to-10"
extractor = NodeFeatureExtractor.get_jaccard_coefficient


res = [f"20-{i}-" + x_label + ".png" for i in idx]
y_label = "frequency"
graphs = [nx.read_graphml(os.path.join(data_dir, graph), node_type=int) for graph in graphs]

if extractor is NodeFeatureExtractor.get_algebraic_distance:
    # (batch_size, graph_size)
    graphs = NodeFeatureExtractor.get_shallow_embed(graphs, 1).squeeze(-1)

# (batch_size, graph_size)
if params is None:
    node_features = extractor(graphs)
else:
    node_features = extractor(graphs, params)

for idx in range(len(graphs)):
    data = node_features[idx, :]
    counts, bins = np.histogram(data)
    plt.hist(bins[:-1], bins, weights=counts, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(res_dir, res[idx]))
    plt.close()
