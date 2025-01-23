import os
from collections import Counter

import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from construct_graph import calc_node_features_init, build_graph_with_centrality, visualize_init_graph, \
    calc_edge_features_init
from get_data import get_processed_data


def train_test_split_and_ros(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    print("INIT distribution of train: ", Counter(y_train))
    print("Distribution of test:", Counter(y_test))

    ros = RandomOverSampler(random_state=42)
    x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)

    print("RESAMPLED distribution of train: ", Counter(y_train_resampled))

    return x_train_resampled, x_test, y_train_resampled, y_test


def generate_sample_graphs(X, G_linear, y=None):
    """
    基于全局图 G_linear 和样本特征值，为每个样本生成样本图 G_i，并保留多边属性。
    同时将标签 y 添加到图中。

    参数：
    --------
    X: pd.DataFrame
        样本特征数据，每行对应一个样本。
    G_linear: networkx.Graph
        全局线性相关图，用于提供初始边结构和多边属性。
    y: np.array or pd.Series, optional
        每个样本的标签。

    返回：
    --------
    sample_graphs: list of Data
        每个样本的图（PyTorch Geometric 格式），包含标签 y。
    """
    # 创建节点名称到整数索引的映射
    node_names = list(G_linear.nodes)
    node_name_to_index = {name: idx for idx, name in enumerate(node_names)}

    sample_graphs = []

    for i in range(X.shape[0]):  # 遍历每个样本
        # 当前样本的特征值
        sample_features = X.iloc[i].values

        # 构建节点特征
        x = torch.tensor([sample_features[node_name_to_index[name]] for name in node_names],
                         dtype=torch.float).unsqueeze(1)

        # 构建边（从 G_linear 中继承）
        edge_index = []
        edge_attr = []

        for (u, v, attrs) in G_linear.edges(data=True):  # 遍历全局图的边
            edge_index.append([node_name_to_index[u], node_name_to_index[v]])  # 将名称映射为整数索引
            edge_attr.append([
                attrs.get('Pearson', 0.0),
                attrs.get('MutualInfo', 0.0),
                attrs.get('CosineSim', 0.0)
            ])

        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # 转为 PyG 格式
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # 转为张量

        # 构建样本图
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # 添加标签
        if y is not None:
            graph.y = torch.tensor(y[i], dtype=torch.long)  # 二分类标签为 long 类型

        sample_graphs.append(graph)

    return sample_graphs


def init_to_gi(raw_data, threshold):
    _, x_norm, y = get_processed_data(raw_data)
    del raw_data
    print(y.value_counts())
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    print(y.value_counts())

    nodes = x_norm.columns
    node_features = calc_node_features_init(x_norm, y)
    edge_features = calc_edge_features_init(x_norm, nodes, y)

    G, percentile = build_graph_with_centrality(x_norm, edge_features, weight_threshold_percentile=threshold)
    visualize_init_graph(
        G,
        layout='circular',
        node_size_key='Degree Centrality',  # 节点大小与中心性相关
        edge_weight_key='weight',  # 使用边的 Pearson 相关系数作为权重
        edge_weight_scale=10,  # 调整边的粗细缩放
        font_size=8,  # 调整字体大小
        figsize=(15, 10),  # 调整图大小
        percentile=percentile  # 储存构图边数
    )

    # G_i construction
    x_train_resampled, x_test, y_train_resampled, y_test = train_test_split_and_ros(x_norm, y)
    x_train_resampled = x_train_resampled.reset_index(drop=True)
    y_train_resampled = y_train_resampled.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("Resampling finished!")

    Gi_train_resampled = generate_sample_graphs(x_train_resampled, G, y_train_resampled)
    Gi_test = generate_sample_graphs(x_test, G, y_test)

    return Gi_train_resampled, Gi_test


if __name__ == "__main__":
    raw_data = os.path.join(os.path.dirname(__file__), 'data', 'nhtsa.csv')
    threshold = 70  # leave (100-threshold) edges

    Gi_train_resampled, Gi_test = init_to_gi(raw_data, threshold)
    print("Data  'Gi'  finished!")
    print(Gi_train_resampled[0])  # 检查第一个样本
