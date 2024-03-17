import os
import networkx as nx
import matplotlib.pyplot as plt

from nets.attention_model import AttentionModel
get_shallow_embed = AttentionModel.get_shallow_embed

data_dir = '../data/barabasi_albert'
res_dir = data_dir
idx = [2, 3]
graphs = [f"20-{i}.graphml" for i in idx]
res = [f"20-{i}-2D.png" for i in idx]
res_together = f"20-{idx[0]}-{idx[-1]}-2D.png"
draw_together = True
draw_label = False

if draw_together:
    fig, ax = plt.subplots()

for i, graph in enumerate(graphs):
    graph = nx.read_graphml(os.path.join(data_dir, graph), node_type=int)
    # (graph_size, node_dim)
    shallow_embed = get_shallow_embed([graph], node_dim=2)[0]
    if not draw_together:
        fig, ax = plt.subplots()

    ax.scatter(shallow_embed[:, 0], shallow_embed[:, 1], marker='o')

    if draw_label:
        for j in range(shallow_embed.size(0)):
            ax.annotate(f"{j}", (shallow_embed[j, 0], shallow_embed[j, 1]))
    if not draw_together:
        plt.savefig(os.path.join(res_dir, res[i]))
        plt.close()

if draw_together:
    plt.savefig(os.path.join(res_dir, res_together))
    plt.close()





