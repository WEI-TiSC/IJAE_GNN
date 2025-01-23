import os

import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score

from get_data import get_processed_data


def calc_node_features_init(X, y):
    """
    仅为普通特征（列）生成节点特征

    参数：
    --------
    X: pd.DataFrame
        特征数据，每列代表一个特征。
    y: pd.Series or np.array
        目标标签（InjurySeverity）。

    返回：
    --------
    node_features: pd.DataFrame
        行索引：X 的特征名
        列：统计信息（Mean, Std, Min, Max, Median, Skewness, Kurtosis, Correlation_with_Target, Importance, Correlation_Sum）
    """
    node_features = pd.DataFrame({
        'Mean': X.mean(),
        'Std': X.std(),
        'Min': X.min(),
        'Max': X.max(),
        'Median': X.median(),
        'Skewness': X.apply(skew),
        'Kurtosis': X.apply(kurtosis)
    })

    # 使用随机森林来衡量特征重要性
    rf = RandomForestClassifier()
    rf.fit(X, y)
    node_features['Importance'] = rf.feature_importances_

    # 计算各特征与其他特征的相关性之和（可作衡量“与全局的耦合度”）
    correlation_matrix = X.corr().abs()
    node_features['Correlation_Sum'] = correlation_matrix.sum(axis=1)

    return node_features


def calc_edge_features_init(X, feature_names, y):
    """
    仅计算特征之间的边特征，不再包含与 'InjurySeverity' 节点的边。

    参数：
    --------
    X: pd.DataFrame
        特征数据，每列代表一个特征。
    feature_names: list
        特征名列表，对应 X 的列顺序。
    y: pd.Series or np.array
        目标标签（InjurySeverity），在这里已不再计算边到目标节点。

    返回：
    --------
    edge_features: dict
        键：(feature_i, feature_j)
        值：一个字典，包括 'Pearson', 'MutualInfo', 'CosineSim' 等
    """
    n_features = X.shape[1]
    edge_features = {}

    # 计算特征之间两两的相关性（Pearson、互信息、余弦相似）
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pearson_corr = X.iloc[:, i].corr(X.iloc[:, j])
            mutual_info = mutual_info_score(X.iloc[:, i], X.iloc[:, j])

            try:
                cosine_sim = 1 - cosine(X.iloc[:, i], X.iloc[:, j])
            except ValueError:
                cosine_sim = float('nan')

            edge_features[(feature_names[i], feature_names[j])] = {
                'Pearson': abs(pearson_corr),
                'MutualInfo': mutual_info,
                'CosineSim': cosine_sim
            }

    return edge_features


def build_graph_with_centrality(
    X,
    edge_feats,
    weight_threshold_percentile=50
):
    """
    基于特征间的 Pearson 相关系数, 构建无 InjurySeverity 节点的图,
    并计算节点及边的中心性信息.

    参数:
    --------
    X : pd.DataFrame
        特征数据, 行为样本, 列为特征.
    edge_feats : dict
        由 (feature_i, feature_j) -> { 'Pearson', 'MutualInfo', 'CosineSim', ... } 组成的字典
        表示特征对之间的各种相关度度量.
    weight_threshold_percentile : float, 默认 50
        用于筛选特征间边的百分位数阈值, 例如 50 表示选取大于该百分位数的 Pearson 才加入图中.

    返回:
    --------
    G : networkx.Graph
        构建好的图, 节点为特征, 带有中心性信息; 边为特征对, 带有相关度信息等.
    weight_threshold_percentile : float
        返回用于本次构图的百分位数阈值, 方便追踪日志/可视化.
    """

    G = nx.Graph()

    # 1. 添加所有特征节点
    features = list(X.columns)
    G.add_nodes_from(features)

    # 2. 收集特征对之间的 Pearson 值, 用于计算阈值
    weights = []
    for (node1, node2), attrs in edge_feats.items():
        if (node1 in features) and (node2 in features):
            weights.append(attrs['Pearson'])
    if len(weights) == 0:
        print("Warning: No valid feature-feature edges found in edge_feats.")
        return G, weight_threshold_percentile

    # 3. 计算权重的百分位阈值
    weight_threshold = np.percentile(weights, weight_threshold_percentile)
    print(f"Selecting top {weight_threshold_percentile}% threshold as: {weight_threshold:.4f}")

    # 4. 添加特征间边(只保留 Pearson >= 阈值的边)
    for (node1, node2), attrs in edge_feats.items():
        if (node1 in features) and (node2 in features):
            if attrs['Pearson'] >= weight_threshold:
                G.add_edge(
                    node1,
                    node2,
                    weight=attrs['Pearson'],
                    MutualInfo=attrs.get('MutualInfo', 0.0),
                    CosineSim=attrs.get('CosineSim', 0.0),
                    Type=0   # 统一边类型 (feat-feat)
                )

    # 5. 计算图的中心性
    degree_centrality = nx.degree_centrality(G)
    pagerank_centrality = nx.pagerank(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # 6. 将中心性指标加入节点属性
    for node in G.nodes:
        G.nodes[node]['Degree Centrality'] = degree_centrality[node]
        G.nodes[node]['PageRank Centrality'] = pagerank_centrality[node]
        G.nodes[node]['Betweenness Centrality'] = betweenness_centrality[node]

    # 7. 为边添加 "节点中心性之和" 属性
    for (node1, node2) in G.edges:
        centrality_sum = (
            G.nodes[node1]['Degree Centrality'] +
            G.nodes[node2]['Degree Centrality']
        )
        G.edges[node1, node2]['CentralitySum'] = centrality_sum

    return G, weight_threshold_percentile


def visualize_init_graph(G, layout='spring', node_size_key=None, edge_weight_key='weight',
                    edge_weight_scale=5, font_size=10, figsize=(15, 10), percentile=90):
    """
    可视化优化后的图

    参数:
    - G: networkx 图对象
    - layout: 布局类型 ('spring', 'circular', 'kamada_kawai', etc.)
    - node_size_key: 用于节点大小的中心性指标键 (如 'Degree Centrality')
    - edge_weight_key: 用于边权重的键 (默认为 'weight')
    - edge_weight_scale: 边权重的缩放因子
    - font_size: 节点标签字体大小
    - figsize: 图的大小
    """
    plt.figure(figsize=figsize)

    # 选择布局
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)  # 弹簧布局
    elif layout == 'circular':
        pos = nx.circular_layout(G)  # 圆形布局
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)  # Kamada-Kawai布局
    else:
        raise ValueError("Unsupported layout type. Choose from 'spring', 'circular', 'kamada_kawai'.")

    # 节点大小
    if node_size_key:
        node_sizes = [G.nodes[node][node_size_key] * 1000 for node in G.nodes]
    else:
        node_sizes = 800  # 默认节点大小

    # 边权重
    edge_weights = [G.edges[edge][edge_weight_key] * edge_weight_scale for edge in G.edges]

    # 边透明度（与权重相关）
    edge_alpha = [0.5 + (G.edges[edge][edge_weight_key] / max(edge_weights)) * 0.5 for edge in G.edges]

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)

    # 绘制节点标签
    nx.draw_networkx_labels(
        G, pos, font_size=font_size, font_color="black", bbox=dict(facecolor='white', edgecolor='none', alpha=0.6)
    )

    # 绘制边
    for i, edge in enumerate(G.edges(data=True)):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(edge[0], edge[1])], width=edge_weights[i], alpha=edge_alpha[i], edge_color='gray'
        )

    # 设置标题
    plt.title(f'Graph with top {100 - percentile}% Pearson', fontsize=16)
    plt.axis('off')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'save_info',
                             'init_graph', f'Graph with top {100 - percentile}% pearson.png'), dpi=300)
    plt.show()


if __name__ == "__main__":
    raw_data = os.path.join(os.path.dirname(__file__), 'data', 'nhtsa.csv')
    x, x_norm, y = get_processed_data(raw_data)
    del raw_data

    threshold = 50

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
    print("\n图的节点信息:")
    for node, attrs in G.nodes(data=True):
        print(f"{node}: {attrs}")

    print("\n图的边信息:")
    for node1, node2, attrs in G.edges(data=True):
        print(f"Edge from {node1} to {node2} with attributes: {attrs}")
