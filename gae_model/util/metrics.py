import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
np.set_printoptions(threshold=np.inf)
def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    '''
    进行评估
    :param emb:经过图卷积的embedding
    :param adj_orig:除去对角元素的邻接矩阵
    :param edges_pos:正样本，有链接关系
    :param edges_neg:负样本，无链接关系
    :return:
    '''

    def sigmoid(inx):
        if inx >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
            return 1.0 / (1 + np.exp(-inx))
        else:
            return np.exp(inx) / (1 + np.exp(inx))

    # predict on val set and test set of edges
    adj_rec = np.dot(emb, emb.T)

    preds = [] #正样本预测值 preds
    pos = [] #正样本真实值 pos
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score