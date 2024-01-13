
import numpy as np

from ge.classify import read_node_label, Classifier
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

from sklearn.metrics import roc_auc_score, average_precision_score
import math
import scipy.sparse as sp
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

adj = nx.read_adjlist("C:\\Users\\王利\\PycharmProjects\\PGAE-Supplement\\data\\mRNA graph(4000) adj.dat")
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
    G=nx.read_adjlist('../data/mRNA graph(4000) adj.dat',
                         create_using = nx.DiGraph(), nodetype = None)
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=4, workers=1, use_rejection_sampling=0)
    model.train(window_size = 5, iter = 3)
    embeddings=model.get_embeddings()

    embs = []
    for emb in embeddings.values():
        embs.append(emb)
    roc_score1, ap_score1 = get_roc_score(test_edges, test_edges_false, embs)
    print("auc_score = " + str(roc_score1))
    print("ap_score = " + str(ap_score1))
