import collections
import random
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def load_graph(path, nodes_path):
    g = nx.read_edgelist(path)
    adj = nx.adjacency_matrix(g)
    adj_triu = sp.triu(adj)
    graph_info = nx.Graph()
    G = collections.defaultdict(dict)

    for i, j in zip(*adj_triu.nonzero()):
        v_i = int(i)
        v_j = int(j)
        i_w = nodes_path.iloc[v_i]
        j_w = nodes_path.iloc[v_j]
        # w = 1.0  # 数据集有权重的话则读取数据集中的权重
        w = np.dot(i_w, j_w)  # 数据集有权重的话则读取数据集中的权重
        G[v_i][v_j] = w
        G[v_j][v_i] = w
        graph_info.add_node(v_i)
        graph_info.add_node(v_j)
        graph_info.add_edge(v_i, v_j)
    return G, graph_info, g.nodes


# 节点类 存储社区与节点编号信息
'''
类中还定义了一个名为Vertex的内部类，该类用于表示原始网络中的节点。
每个Vertex实例都有一个编号、一个社区编号和一个包含与其相连边的权重值的字典。
'''


class Vertex:
    def __init__(self, vid, cid, nodes, k_in=0):
        # 节点编号
        self._vid = vid
        # 社区编号
        self._cid = cid
        self._nodes = nodes
        self._kin = k_in  # 结点内部的边的权重


class Louvain:
    def __init__(self, G):
        self._G = G
        self._m = 0  # 边数量 图会凝聚动态变化
        self._cid_vertices = {}  # 需维护的关于社区的信息(社区编号,其中包含的结点编号的集合)
        self._vid_vertex = {}  # 需维护的关于结点的信息(结点编号，相应的Vertex实例)
        tol_weigth = 0
        edges = 0
        for vid, weigth in zip(self._G.keys(), self._G.values()):
            # 刚开始每个点作为一个社区
            self._cid_vertices[vid] = {vid}
            # 刚开始社区编号就是节点编号
            self._vid_vertex[vid] = Vertex(vid, vid, {vid})
            # 计算边数  每两个点维护一条边
            self._m += sum([1 for neighbor in self._G[vid].keys()
                            if neighbor > vid])
            edges += len(weigth)
            tol_weigth += sum(weigth.values())
        self._avg_weigth = tol_weigth / edges

    # 模块度优化阶段
    '''
    类中定义了一个名为first_stage的函数，该函数执行Louvain算法的第一阶段，即模块度优化阶段。
    该函数在每个节点的邻居之间进行遍历，并计算将节点加入每个邻居的社区会带来的模块度增益。
    如果某个社区的模块度增益大于0，则将节点加入该社区。
    '''

    def first_stage(self):
        mod_inc = False  # 用于判断算法是否可终止
        visit_sequence = [v_vid for v_vid, v in self._G.items() if sum(v.values()) >= self._avg_weigth]
        # visit_sequence = self._G.keys()  # 是一个列表，存储了所有节点的编号，用于在第一阶段中遍历所有节点。
        # 随机访问
        random.shuffle(list(visit_sequence))
        max_iter = 100  # set a maximum number of iterations
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
        # while True:
        # for i in range(10):  # 这个无限循环的作用是不断进行社区划分的调整，直到无法继续增加社区模块度为止。
            can_stop = True  # 第一阶段是否可终止
            # 遍历所有节点
            for v_vid in visit_sequence:  # 遍历所有节点，并对每个节点进行社区划分的调整
                '''
                v_cid 是当前节点 
                v_vid 所在的社区编号
                k_v 是当前节点 v_vid 的权重，即度数。这里的度数是指节点的内部边权重之和加上外部边权重之和。
                cid_Q 是一个字典，用于存储将当前节点 v_vid 加入到每个社区后，社区模块度的增益。
                '''
                # 获得节点的社区编号
                v_cid = self._vid_vertex[v_vid]._cid
                # k_v节点的权重(度数)  内部与外部边权重之和
                k_v = sum(self._G[v_vid].values()) + self._vid_vertex[v_vid]._kin
                # 存储模块度增益大于0的社区编号
                cid_Q = {}
                # 遍历节点的邻居,计算将当前节点 v_vid 加入到每个社区后，社区模块度的增益。
                '''
                首先计算社区 w_cid 中所有节点的链路上的权重的总和 tot。
                如果当前节点 v_vid 就在社区 w_cid 中，那么要从 tot 中减去当前节点的权重 k_v，因为当前节点的权重已经包含在 tot 中。
                然后计算当前节点 v_vid 连接到社区 w_cid 中的节点的链路的总和 k_v_in。
                最后计算增益 delta_Q，并将增益和社区编号 w_cid 存储在字典 cid_Q 中。
                注意，这里的增益是指将当前节点 v_vid 加入到社区 w_cid 中后，社区模块度的变化量。
                '''

                for w_vid in self._G[v_vid].keys():
                    # 获得该邻居的社区编号
                    w_cid = self._vid_vertex[w_vid]._cid
                    if w_cid in cid_Q:
                        continue
                    else:
                        # tot是关联到社区C中的节点的链路上的权重的总和
                        tot = sum(
                            [sum(self._G[k].values()) + self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]])
                        if w_cid == v_cid:
                            tot -= k_v
                        # k_v_in是从节点i连接到C中的节点的链路的总和
                        k_v_in = sum(
                            [v for k, v in self._G[v_vid].items() if k in self._cid_vertices[w_cid]])
                        # 由于只需要知道delta_Q的正负，所以少乘了1/(2*self._m)
                        delta_Q = k_v_in - k_v * tot / self._m
                        cid_Q[w_cid] = delta_Q

                # 取得最大增益的编号 将字典 cid_Q 按照增益从大到小排序，并取出增益最大的社区编号 cid 和增益 max_delta_Q
                cid, max_delta_Q = sorted(
                    cid_Q.items(), key=lambda item: item[1], reverse=True)[0]
                # 如果增益 max_delta_Q 大于 0，且当前节点的原来的社区编号 v_cid 不等于增益最大的社区编号 cid，则表示需要将当前节点 v_vid 加入到增益最大的社区 cid 中。
                if max_delta_Q > 0.0 and cid != v_cid:
                    # 让该节点的社区编号变为取得最大增益邻居节点的编号
                    self._vid_vertex[v_vid]._cid = cid
                    # 在该社区编号下添加该节点
                    self._cid_vertices[cid].add(v_vid)
                    # 以前的社区中去除该节点
                    self._cid_vertices[v_cid].remove(v_vid)
                    # 模块度还能增加 继续迭代
                    can_stop = False
                    mod_inc = True
            if can_stop:
                break
        return mod_inc

    # 网络凝聚阶段
    '''
    类中还定义了一个名为second_stage的函数，该函数执行Louvain算法的第二阶段，即社区缩减阶段。
    该函数创建了一个新的网络，其中的节点表示原始网络中的社区，边表示原始网络中的边。
    然后，该函数对新的网络重新执行第一阶段，直到不再有节点可以转移到其他社区为止。
    '''

    def second_stage(self):

        # 1 创建新的社区和节点。在这一部分中，对于每一个社区，会遍历该社区内的所有节点，然后将这些节点看做一个节点。
        cid_vertices = {}
        vid_vertex = {}
        # 遍历社区和社区内的节点
        for cid, vertices in self._cid_vertices.items():
            if len(vertices) == 0:
                continue
            new_vertex = Vertex(cid, cid, set())
            # 将该社区内的所有点看做一个点
            for vid in vertices:
                new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
                new_vertex._kin += self._vid_vertex[vid]._kin
                # k,v为邻居和它们之间边的权重 计算kin社区内部总权重 这里遍历vid的每一个在社区内的邻居   因为边被两点共享后面还会计算  所以权重/2
                for k, v in self._G[vid].items():
                    if k in vertices:
                        new_vertex._kin += v / 2.0
            # 新的社区与节点编号
            cid_vertices[cid] = {cid}
            vid_vertex[cid] = new_vertex

        # 2 计算社区之间边的权重。在这一部分中，会遍历所有的社区，然后计算两个社区之间边的权重。
        G = collections.defaultdict(dict)
        # 遍历现在不为空的社区编号 求社区之间边的权重
        for cid1, vertices1 in self._cid_vertices.items():
            if len(vertices1) == 0:
                continue
            for cid2, vertices2 in self._cid_vertices.items():
                # 找到cid后另一个不为空的社区
                if cid2 <= cid1 or len(vertices2) == 0:
                    continue
                edge_weight = 0.0
                # 遍历 cid1社区中的点
                for vid in vertices1:
                    # 遍历该点在社区2的邻居已经之间边的权重(即两个社区之间边的总权重  将多条边看做一条边)
                    for k, v in self._G[vid].items():
                        if k in vertices2:
                            edge_weight += v
                if edge_weight != 0:
                    G[cid1][cid2] = edge_weight
                    G[cid2][cid1] = edge_weight
        # 3 更新社区和节点。在这一部分中，会将当前的社区和节点更新为新的社区和节点，并更新图中的边。
        self._cid_vertices = cid_vertices
        self._vid_vertex = vid_vertex
        self._G = G

    # 获取聚类的结果。它遍历所有的社区，然后将社区内的所有节点加入到一个集合中，最后将这个集合转换成一个列表，并将这个列表添加到结果列表中
    def get_communities(self):
        communities = []
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(list(c))
        return communities

    def execute(self):
        # iter_time = 1
        while True:
            # iter_time += 1
            # 反复迭代，直到网络中任何节点的移动都不能再改善总的 modularity 值为止
            mod_inc = self.first_stage()
            if mod_inc:
                self.second_stage()
            else:
                break
        return self.get_communities()


# 可视化划分结果

def showCommunity(G, partition, pos, g_nodes):
    # 划分在同一个社区的用一个符号表示，不同社区之间的边用黑色粗体
    cluster = {}
    labels = {}
    g_nodes = list(g_nodes)
    for index, item in enumerate(partition):
        for nodeID in item:
            labels[nodeID] = r'$' + str(g_nodes[nodeID]) + '$'  # 设置可视化label
            cluster[nodeID] = index  # 节点分区号 key是节点值，value是社区号

    # 可视化节点
    colors = [f'#{random.randint(0, 0xff):02x}{random.randint(0, 0xff):02x}{random.randint(0, 0xff):02x}' for _ in
              range(len(G))]
    shapes = ['v', 'D', 'o', '^', '<']
    '''
    'o'：圆形
    's'：正方形
    '^'：三角形
    'd'：菱形
    'p'：五边形
    'h'：六边形
    '''
    for index, item in enumerate(partition):
        nx.draw_networkx_nodes(G, pos, nodelist=item,
                               node_color=random.choice(colors),
                               node_shape=random.choice(shapes),
                               node_size=200,
                               alpha=1)

    # 可视化边
    edges = {len(partition): []}
    # # print(cluster)
    # # print(G.edges())
    for link in G.edges():
        # cluster间的link
        if cluster[link[0]] != cluster[link[1]]:
            edges[len(partition)].append(link)
        else:
            # cluster内的link
            if cluster[link[0]] not in edges:
                edges[cluster[link[0]]] = [link]
            else:
                edges[cluster[link[0]]].append(link)
    print(edges)
    '''
    在这段代码中，字典的键为 len(partition)，它表示社区间的边。
    对于其他键，它们表示社区内的边，并且键的值表示社区的分区号。
    
    当遍历字典中的边时，如果字典的键小于 len(partition)，则表示这是社区内的边，否则表示这是社区间的边。
     在这种情况下，社区内的边会使用细线绘制，社区间的边会使用粗线绘制。
    # '''
    # for index, edgelist in enumerate(edges.values()):
    #     # cluster内
    #     if index < len(partition):
    #         nx.draw_networkx_edges(G, pos,
    #                                edgelist=edgelist,
    #                                width=1, alpha=0.8, edge_color=random.choice(colors))
    #     else:
    #         # cluster间
    #         nx.draw_networkx_edges(G, pos,
    #                                edgelist=edgelist,
    #                                width=3, alpha=0.8, edge_color=random.choice(colors))
    #
    # # 可视化label，节点名字
    # nx.draw_networkx_labels(G, pos, labels, font_size=10)
    plt.axis('off')
    plt.show()


def showSingleCommunity(G, partition, g_nodes, count, f):
    # 划分在同一个社区的用一个符号表示，不同社区之间的边用黑色粗体
    cluster = {}
    g_nodes = list(g_nodes)
    for item in partition:
        cluster[item] = count  # 节点分区号 key是节点值，value是社区号
    edges = {count: []}
    for link in G.edges():
        # cluster内的link
        if link[0] in cluster and link[1] in cluster:
            #检查这个端点所在的社区是否在 edges 字典中
            if cluster[link[0]] not in edges:
                edges[cluster[link[0]]] = [link]
            else:
                edges[cluster[link[0]]].append(link)
            if str(g_nodes[link[0]]).startswith("MGG") or str(g_nodes[link[0]]).startswith("srna"):
                inter = str(g_nodes[link[0]]) + '\t' + str(g_nodes[link[1]]) + '\t' + str(count) + '\n'
            else:
                inter = str(g_nodes[link[1]]) + '\t' + str(g_nodes[link[0]]) + '\t' + str(count) + '\n'
            f.write(inter)

'''
该函数首先使用输入的图G的边数m来计算出网络的总连边数。

接下来，使用两个列表a、e来存储每个社区的内部边的数量和每个社区的边界边的数量。

对于每个社区，遍历它的所有节点，计算出节点的度数，并将所有节点的度数加起来除以2 * m，得到的结果是该社区的内部边的数量a。

然后，对于每个社区，遍历它的所有节点对，如果它们之间有边，则将边计数器加1，最后将边计数器除以2 * m，得到的结果是该社区的边界边的数量e。

最后，使用内部边的数量a和边界边的数量e计算Q指标，将每个社区的贡献加起来，并返回Q指标的总和。
'''
def cal_Q(partition, G):  # 计算Q
    m = len(G.edges(None, False))
    a = []
    e = []
    for community in partition:  # 把每一个联通子图拿出来
        t = 0.0
        for node in community:  # 找出联通子图的每一个顶点
            # G.neighbors(node)找node节点的邻接节点
            # t += len([x for x in G.neighbors(node)])
            t += len(list(G.neighbors(node)))
        a.append(t / (2 * m))
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q




if __name__ == '__main__':
    pass
