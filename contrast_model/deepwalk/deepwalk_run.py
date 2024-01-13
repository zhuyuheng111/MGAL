
import numpy as np
import math
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score, average_precision_score

from deepwalk import DeepWalk

import networkx as nx
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

adj = nx.read_adjlist("/home/yuanshuai20/paper/互作对.txt")
# adj = nx.read_adjlist("/home/yuanshuai20/paper/基因组互作对.txt")
# adj = nx.read_adjlist("/home/yuanshuai20/paper/转录组互作对.txt")
# adj = nx.read_adjlist("/home/yuanshuai20/paper/蛋白组互作对.txt")
print(adj)
adj = nx.adj_matrix(adj)

adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

def get_roc_score(edges_pos, edges_neg, embeddings):

    def fun(x):
        return 1 / (1 + x)

    def dist(emb1, emb2):
        distance = 0.0
        for i in range(len(emb1)):
            distance += math.pow(emb1[i] - emb2[i],2)
        return math.sqrt(distance)

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        metric = fun(dist(embeddings[e[0]],embeddings[e[1]]))
        preds.append(metric)
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        metric = fun(dist(embeddings[e[0]], embeddings[e[1]]))
        preds_neg.append(metric)
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


if __name__ == "__main__":
    G = nx.read_adjlist('/home/yuanshuai20/paper/互作对.txt',
    # G = nx.read_adjlist('/home/yuanshuai20/paper/基因组互作对.txt',
    # G = nx.read_adjlist('/home/yuanshuai20/paper/转录组互作对.txt',
    # G = nx.read_adjlist('/home/yuanshuai20/paper/蛋白组互作对.txt',
                         create_using=nx.Graph(), nodetype=None)

    model = DeepWalk(G, walk_length=40, num_walks=10)
    model.train()
    embeddings = model.get_embeddings()

    embs = []
    for emb in embeddings.values():
        embs.append(emb)
    roc_score1, ap_score1 = get_roc_score(test_edges, test_edges_false, embs)
    print("auc_score = " + str(roc_score1))
    print("ap_score = " + str(ap_score1))

