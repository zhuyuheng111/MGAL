import collections
import numpy as np
import networkx as nx
import scipy.sparse as sp



def load_graph(matrix_file, nodes_path):
    #读取边，创建邻接矩阵，得到上三角
    g = nx.read_edgelist(matrix_file)
    adj = nx.adjacency_matrix(g)
    adj_triu = sp.triu(adj)
    graph_info = nx.Graph()

    for i, j in zip(*adj_triu.nonzero()):
        v_i = int(i)
        v_j = int(j)
        i_w = nodes_path.iloc[v_i]
        j_w = nodes_path.iloc[v_j]
        w = np.dot(i_w, j_w)  # 数据集有权重的话则读取数据集中的权重
        graph_info.add_node(v_i)
        graph_info.add_node(v_j)
        graph_info.add_weighted_edges_from([(v_i, v_j,w)])
    return graph_info