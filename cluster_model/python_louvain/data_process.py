import collections
import numpy as np
import networkx as nx
import scipy.sparse as sp


def load_graph(matrix_file, nodes_path):
    # 读取边，创建邻接矩阵，得到上三角
    g = nx.read_edgelist(matrix_file)
    adj = nx.adjacency_matrix(g)
    adj_triu = sp.triu(adj)
    graph_info = nx.Graph()

    for i, j in zip(*adj_triu.nonzero()):
        v_i = int(i)
        v_j = int(j)
        i_w = nodes_path.iloc[v_i]
        j_w = nodes_path.iloc[v_j]
        w = abs(np.dot(i_w, j_w))  # 数据集有权重的话则读取数据集中的权重
        # w = abs(np.dot(i_w, j_w)) / (np.linalg.norm(i_w) * np.linalg.norm(j_w))
        graph_info.add_node(v_i)
        graph_info.add_node(v_j)
        graph_info.add_weighted_edges_from([(v_i, v_j, w)])
    return graph_info, g.nodes


def SingleCommunity(communitie, graph, g_nodes, count, f):
    g_nodes = list(g_nodes)
    for u, v in graph.edges():
        if u in communitie and v in communitie:
            if str(g_nodes[u]).startswith("MGG") or str(g_nodes[u]).startswith("srna"):
                inter = str(g_nodes[u]) + '\t' + str(g_nodes[v]) + '\t' + str(count) + '\n'
            else:
                inter = str(g_nodes[v]) + '\t' + str(g_nodes[u]) + '\t' + str(count) + '\n'
            f.write(inter)


def PortraitCommunity(communitie, g_nodes, count, f):
    g_nodes = list(g_nodes)
    for i in range(len(communitie)):
        for j in range(i + 1, len(communitie)):
            if (str(g_nodes[communitie[i]]).startswith("M") and str(g_nodes[communitie[j]]).startswith("srna")) or \
                (str(g_nodes[communitie[i]]).startswith("srna") and str(g_nodes[communitie[j]]).startswith("M")) or \
                (str(g_nodes[communitie[i]]).startswith("M") and "T" in str(g_nodes[communitie[i]]) and str(g_nodes[communitie[j]]).startswith("M") and not "T" in str(g_nodes[communitie[j]])) or \
                (str(g_nodes[communitie[i]]).startswith("M") and not "T" in str(g_nodes[communitie[i]]) and str(g_nodes[communitie[j]]).startswith("M") and "T" in str(g_nodes[communitie[j]])):
                inter = str(g_nodes[communitie[i]]) + '\t' + str(g_nodes[communitie[j]]) + '\t' + str(count) + '\n'
                f.write(inter)
