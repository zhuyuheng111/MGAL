import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, KernelPCA, MiniBatchSparsePCA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from umap import UMAP
from sklearn.utils.extmath import randomized_svd
import numpy as np

warnings.filterwarnings("ignore")
plt.style.use('bmh')


# a simple timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        run_time = end - start
        return run_time, result

    return wrapper


# svm to quickly test out the results
@timer
def do_pca(data, n_components):
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(data)
    return result


@timer
def do_ica(data, n_components):
    ica = FastICA(n_components=n_components)
    result = ica.fit_transform(data)
    return result


@timer
def do_factor_analysis(data, n_components):
    fa = FactorAnalysis(n_components=n_components)
    result = fa.fit_transform(data)
    return result


@timer
def do_kpca(data, n_components, kernel):
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    result = kpca.fit_transform(data)
    return result


@timer
def do_minibatch_sparse_pca(data, n_components):
    mbs_pca = MiniBatchSparsePCA(n_components=n_components)
    result = mbs_pca.fit_transform(data)
    return result


@timer
def do_svd(data, n_components):
    svd = TruncatedSVD(n_components=n_components)
    result = svd.fit_transform(data)
    return result


@timer
def do_tsne(data, n_components):
    tsne = TSNE(n_components=n_components)
    result = tsne.fit_transform(data)
    return result


@timer
def do_isomap(data, n_components):
    iso = Isomap(n_components=n_components)
    result = iso.fit_transform(data)
    return result


@timer
def do_lle(data, n_components):
    lle = LocallyLinearEmbedding(n_components=n_components)
    result = lle.fit_transform(data)
    return result


@timer
def do_mds(data, n_components):
    mds = MDS(n_components=n_components)
    result = mds.fit_transform(data)
    return result


@timer
def do_umap(data, n_components):
    umap = UMAP(n_components=n_components)
    result = umap.fit_transform(data)
    return result
@timer
def do_tsvd(data, n_components):
    u, s, vt = randomized_svd(data, n_components)
    result = np.dot(u, np.diag(s))
    return result

def run_dimension(data, n_components):
    # print("Start running PCA...")
    # run_time_pca, result_pca = do_pca(data, n_components)
    # print(run_time_pca)
    # print("Start running ICA...")
    # run_time_ica, result_ica = do_ica(data, n_components)
    # print(run_time_ica)
    # print("Start running Factor Analysis...")
    # run_time_fa, result_fa = do_factor_analysis(data, n_components)
    # print(run_time_fa)
    # print("Start running KPCA...")
    # run_time_kpca, result_kpca = do_kpca(data, n_components, "rbf")
    # print(run_time_kpca)
    # print("Start running Mini-Batch Sparse PCA...")
    # run_time_mbspca, result_mbspca = do_minibatch_sparse_pca(data, n_components)
    # print(run_time_mbspca)
    # print("Start running SVD...")
    # run_time_svd, result_svd = do_svd(data, n_components)
    # print(run_time_svd)
    # print("Start running Isomap...")
    # run_time_isomap, result_isomap = do_isomap(data, n_components)
    # print(run_time_isomap)
    # print("Start running LLE...")
    # run_time_lle, result_lle = do_lle(data, n_components)
    # print(run_time_lle)
    print("Start running MDS...")
    run_time_mds, result_mds = do_mds(data, n_components)
    print(run_time_mds)
    # print("Start running t-SNE...")
    # run_time_tsne, result_tsne = do_tsne(data, n_components)
    # print(run_time_tsne)
    # print("Start running UMAP...")
    # run_time_umap, result_umap = do_umap(data, n_components)
    # print(run_time_umap)
    # print("Start running t-SVD...")
    # run_time_tsvd, result_tsvd = do_tsvd(data, n_components)
    # print(run_time_umap)
    return {
        # "run_time_pca": run_time_pca,
        # "result_pca": result_pca,
        # "run_time_ica": run_time_ica,
        # "result_ica": result_ica,
        # "run_time_fa": run_time_fa,
        # "result_fa": result_fa,
        # "run_time_kpca": run_time_kpca,
        # "result_kpca": result_kpca,
        # "run_time_mbspca": run_time_mbspca,
        # "result_mbspca": result_mbspca,
        # "run_time_svd": run_time_svd,
        # "result_svd": result_svd,
        # "run_time_isomap": run_time_isomap,
        # "result_isomap": result_isomap,
        # "run_time_lle": run_time_lle,
        # "result_lle": result_lle,
        "run_time_mds": run_time_mds,
        "result_mds": result_mds,
        # "run_time_tsne": run_time_tsne,
        # "result_tsne": result_tsne,
        # "run_time_umap": run_time_umap,
        # "result_umap": result_umap,
        # # "run_time_tsvd": run_time_tsvd,
        # # "result_tsvd": result_tsvd,
    }


if __name__ == '__main__':
    print('开始降维')
    emb = pd.read_csv('/home/yuanshuai20/paper/互作对.csv', index_col=0)
    data = emb.values
    index = emb.index
    dismension = run_dimension(data, 2)
    for name, value in dismension.items():
        if "time" not in name:
            filename = f"{name.split('_')[1]}.csv"
            df = pd.DataFrame(value, index=index)
            df.to_csv('/home/yuanshuai20/paper/' + filename)
    print('降维结束')