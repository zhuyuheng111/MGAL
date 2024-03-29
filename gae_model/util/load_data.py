import numpy as np
import scipy.sparse as sp
import networkx as nx

def load_data(sample_data_path):
    '''
    加载邻接矩阵，这里利用networkx读取文件，生成图和邻接矩阵
    生成的节点的编号是根据节点在文件中出现的顺序进行编号
    :param sample_data_path:
    :return:
    '''
    g = nx.read_edgelist(sample_data_path)
    # g = nx.read_weighted_edgelist(sample_data_path)
    adj = nx.adjacency_matrix(g)
    return adj,g.nodes

# 这个函数的作用就是 返回一个稀疏矩阵的非0值坐标、非0值和整个矩阵的shape
###三元组 第一个是坐标 第二是坐标对应值 第三个就是shape
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):  # 判断是否为coo_matrix类型
        sparse_mx = sparse_mx.tocoo()     # 返回稀疏矩阵的coo_matrix形式
    # 这个coo_matrix类型 其实就是系数矩阵的坐标形式：（所有非0元素 （row，col））根据row和col对应的索引对应非0元素在矩阵中的位置
    # 其他位置自动补0
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    # vstack 按垂直方向排列 再转置 则每一行的两个元素组成的元组就是非0元素的坐标
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    '''
    :param adj:
    :return:
    '''
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0]) # 邻接矩阵加入自身信息，adj = adj + I
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten()) # 节点的度矩阵
    # 正则化，D^{-0.5}(adj+I)D^{-0.5}
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj):
    '''

    '''
    '''
    triu取出给定矩阵的上三角部分，
    sparse_to_tuple函数将其转换为三元组形式，
    edges_all获取图中所有边的坐标信息
    '''
    adj_triu = sp.triu(adj) # 取出稀疏矩阵的上三角部分的非零元素，返回的是coo_matrix类型
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0] # 取除去节点自环的所有边（注意，由于adj_tuple仅包含原始邻接矩阵上三角的边，所以edges中的边虽然只记录了边<src,dis>，而不冗余记录边<dis,src>），shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    edges_all = sparse_to_tuple(adj)[0] # 取原始graph中的所有边，shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号

    '''
    函数计算出测试集和验证集中应包含的边的数量，并使用 range 函数创建一个整数序列，表示所有边的编号。
    接着，使用 shuffle 函数打乱这个序列的顺序，然后取前 num_val 个作为验证集的编号，取前 num_val + num_test 个作为测试集的编号。
    接着，函数使用 delete 函数从所有边中删除测试集和验证集的边，剩余的边就是训练集的边。
    '''
    num_test = int(np.floor(edges.shape[0]*0.2))
    num_val = int(np.floor(edges.shape[0]*0.2))
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx) # 打乱all_edge_idx的顺序
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx] # edges是除去节点自环的所有边（因为数据集中的边都是无向的，edges只是存储了<src,dis>,没有存储<dis,src>），shape=(边数,2)每一行记录一条边的起始节点和终点节点的编号
    val_edges = edges[val_edge_idx]
    # np.vstack():垂直方向堆叠，np.hstack()：水平方向平铺
    # 删除test和val数据集，留下train数据集
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)


    def ismemeber(a, b):
        #判断随机生成的<a,b>这条边是否是已经真实存在的边，如果是，则返回True，否则返回False
        rows_close = np.all((a - b[:, None]) == 0, axis=-1)
        return np.any(rows_close)
    '''
    循环来生成新的测试边,测试边并不一定真实存在于原始的图中，也就是说它们可能是“假”的边
    '''
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        # test集中生成负样本边，即原始graph中不存在的边
        n_rnd = len(test_edges) - len(test_edges_false)
        # 随机生成
        rnd = np.random.randint(0, adj.shape[0], size=2*n_rnd)
        idxs_i = rnd[:n_rnd]
        idxs_j = rnd[n_rnd:]
        for i in range(n_rnd):
            idx_i = idxs_i[i]
            idx_j = idxs_j[i]
            if idx_i == idx_j:
                continue
            if ismemeber([idx_i, idx_j], edges_all): # 如果随机生成的边<idx_i,idx_j>是原始graph中真实存在的边
                continue
            if test_edges_false: # 如果test_edges_false不为空
                if ismemeber([idx_j, idx_i], np.array(test_edges_false)): # 如果随机生成的边<idx_j,idx_i>是test_edges_false中已经包含的边
                    continue
                if ismemeber([idx_i, idx_j], np.array(test_edges_false)): # 如果随机生成的边<idx_i,idx_j>是test_edges_false中已经包含的边
                    continue
            test_edges_false.append([idx_i, idx_j])

    val_edge_false = []
    while len(val_edge_false) < len(val_edges):
        # val集中生成负样本边，即原始graph中不存在的边
        n_rnd = len(val_edges) - len(val_edge_false)
        rnd = np.random.randint(0, adj.shape[0], size=2*n_rnd)
        idxs_i = rnd[:n_rnd]
        idxs_j = rnd[n_rnd:]
        for i in range(n_rnd):
            idx_i = idxs_i[i]
            idx_j = idxs_j[i]
            if idx_i == idx_j:
                continue
            if ismemeber([idx_i, idx_j], train_edges):
                continue
            if ismemeber([idx_j, idx_i], train_edges):
                continue
            if ismemeber([idx_i, idx_j], val_edges):
                continue
            if ismemeber([idx_j, idx_i], val_edges):
                continue
            if val_edge_false:
                if ismemeber([idx_j, idx_i], np.array(val_edge_false)):
                    continue
                if ismemeber([idx_i, idx_j], np.array(val_edge_false)):
                    continue
            val_edge_false.append([idx_i, idx_j])

    # re-build adj matrix
    # data = np.ones(train_edges.shape[0])

    data = []
    for i in train_edges:
        data.append(adj[i[0],i[1]])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    # 这些边列表只包含一个方向的边（adj_train是矩阵，不是edge lists）

    return adj_train, train_edges, val_edges, val_edge_false, test_edges, test_edges_false





