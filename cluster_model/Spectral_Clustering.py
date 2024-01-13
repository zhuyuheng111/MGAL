from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from  data_process import load_graph
# 使用上面的函数加载图
graph = load_graph(matrix_file, nodes_path)

# 从图中创建邻接矩阵
adj_matrix = nx.to_numpy_matrix(graph)

# 定义要测试的簇数范围
range_n_clusters = range(2, 11)

# 创建一个空列表来存储轮廓系数
silhouette_scores = []

# 遍历簇数
for n_clusters in range_n_clusters:
    # 创建 SpectralClustering 对象
    spectral_clustering = SpectralClustering(n_clusters=n_clusters)
    # 将模型拟合到邻接矩阵上
    spectral_clustering.fit(adj_matrix)
    # 获取每个节点的簇标签
    node_labels = spectral_clustering.labels_
    # 计算当前簇数的轮廓系数
    silhouette_score_ = silhouette_score(adj_matrix, node_labels)
    # 将轮廓系数添加到列表中
    silhouette_scores.append(silhouette_score_)

# 绘制轮廓系数与簇数之间的关系图
plt.plot(range_n_clusters, silhouette_scores)
plt.xlabel('簇数')
plt.ylabel('轮廓系数')
plt.show()
