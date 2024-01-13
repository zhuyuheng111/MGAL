import pandas as pd
import os
import sys
from configparser import ConfigParser
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

sys.path.append("/home/yuanshuai20/paper/cluster_model/")
from data_process import load_graph
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.unicode.east_asian_width', True) #设置输出右对齐
class Train():
    def __init__(self):
        pass

    def train_model(self, config_path):
        if os.path.exists(config_path) and (os.path.split(config_path)[1].split('.')[0] == 'config') and (
                os.path.splitext(config_path)[1].split('.')[1] == 'cfg'):
            # load config file
            config = ConfigParser()
            config.read(config_path)
            section = config.sections()[0]
            # data catalog path
            matrix_file = config.get(section, "matrix_file")
            # gae_model save/load path
            nodes_path = config.get(section, "nodes_path")
            # cluster_path = config.get(section, "cluster_path")
            # gae_model param config
        nodes_path = pd.read_csv(nodes_path,index_col=0)
        graph = load_graph(matrix_file,nodes_path)
        # 从图中创建邻接矩阵
        adj_matrix = nx.to_numpy_matrix(graph)

        # 定义要测试的簇数范围
        range_n_clusters = range(2, 61)

        # 创建一个空列表来存储轮廓系数
        silhouette_scores = []

        # 遍历簇数
        for n_clusters in range_n_clusters:
            # 创建 SpectralClustering 对象
            spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize')
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


if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(), 'config.cfg')
    train = Train()
    train.train_model(config_path)
