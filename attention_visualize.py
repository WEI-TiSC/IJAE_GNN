import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from G_linear_to_G_i import init_to_gi
from Gi_GAT_learning import GATModel
from focal_loss import FocalLoss


# Step 1: Modify Attention Extraction Function
def extract_attention_weights(model, data_loader, device, mode="global", class_label=None, target_node=None):
    edge_weights = {}
    node_classes = {}

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            for layer in model.gat_layers:
                attn_output = layer(batch.x, batch.edge_index, batch.edge_attr, return_attention_weights=True)

                if isinstance(attn_output, tuple) and len(attn_output) == 2:
                    edge_index, attn_weights = attn_output
                else:
                    raise ValueError("Unexpected structure of attn_output!")

                # 如果 attn_weights 是一个元组，逐个处理内部张量
                if isinstance(attn_weights, tuple):
                    attn_weights_list = [w.cpu() for w in attn_weights]
                    # 检查并调整张量形状
                    shapes = [w.shape for w in attn_weights_list]
                    if len(set(shapes)) > 1:
                        raise ValueError("Tensor shapes do not match: {}".format(shapes))
                    attn_weights = torch.cat(attn_weights_list, dim=0)
                else:
                    attn_weights = attn_weights.cpu()  # 如果不是元组，直接转为 CPU

                edges = edge_index.t().cpu().numpy()
                weights = attn_weights.numpy()

                for edge, weight in zip(edges, weights):
                    u, v = edge
                    edge = tuple(sorted((u, v)))
                    edge_weights[edge] = edge_weights.get(edge, []) + [weight]

                # Store node predictions for class-specific and local modes
                if mode in ["class_specific", "local"]:
                    preds = model(batch.x, batch.edge_index, batch.edge_attr, batch=batch.batch).argmax(dim=1)
                    for node, cls in zip(batch.batch.cpu().numpy(), preds.cpu().numpy()):
                        node_classes[node] = cls

    edge_weights = {k: sum(v) / len(v) for k, v in edge_weights.items()}

    if mode == "class_specific":
        if class_label is None:
            raise ValueError("class_label must be specified for 'class_specific' mode.")
        edge_weights = {
            edge: weight
            for edge, weight in edge_weights.items()
            if node_classes.get(edge[0]) == class_label or node_classes.get(edge[1]) == class_label
        }

    if mode == "local":
        if target_node is None:
            raise ValueError("target_node must be specified for 'local' mode.")
        edge_weights = {
            edge: weight
            for edge, weight in edge_weights.items()
            if target_node in edge
        }

    return edge_weights

# Step 2: Modify Graph Construction and Visualization Functions
def build_graph(edge_weights, percentile=None):
    G = nx.Graph()
    weights = list(edge_weights.values())

    threshold = np.percentile(weights, percentile) if percentile else None
    for (u, v), w in edge_weights.items():
        if threshold is None or w >= threshold:
            G.add_edge(u, v, weight=w)
    return G


def visualize_graph(G, title, save_path):
    pos = nx.spring_layout(G, seed=42)
    edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="gray", alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(title)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{title.replace(' ', '_')}.png", dpi=300)
    plt.show()


# Step 3: Integrate Analysis into Training Code
if __name__ == "__main__":
    raw_data = os.path.join(os.path.dirname(__file__), 'data', 'nhtsa.csv')
    threshold = 70
    batch_size = 512
    epochs = 3
    input_dim = 1
    hidden_dim = 128
    output_dim = 2
    edge_dim = 3
    num_layers = 5
    dropout = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Gi_train_resampled, Gi_test = init_to_gi(raw_data, threshold)

    train_loader = DataLoader(Gi_train_resampled, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Gi_test, batch_size=batch_size, shuffle=False)

    model = GATModel(input_dim, hidden_dim, output_dim, edge_dim, num_layers=num_layers, dropout=dropout).to(
        device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    class_weights = torch.tensor([1.0, 20.0], dtype=torch.float).to(device)
    criterion = FocalLoss(alpha=1, gamma=4, weight=class_weights)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

    edge_weights_global = extract_attention_weights(model, test_loader, device, mode="global")
    G_global = build_graph(edge_weights_global, percentile=90)
    visualize_graph(G_global, "Global Attention Graph", "visualizations")

    edge_weights_class = extract_attention_weights(model, test_loader, device, mode="class_specific", class_label=1)
    G_class = build_graph(edge_weights_class)
    visualize_graph(G_class, "Class-Specific Attention Graph", "visualizations")

    edge_weights_local = extract_attention_weights(model, test_loader, device, mode="local", target_node=5)
    G_local = build_graph(edge_weights_local)
    visualize_graph(G_local, "Local Attention Graph", "visualizations")
