import os
import sys
from configparser import ConfigParser
from community import community_louvain
import pandas as pd
import torch
from networkx.algorithms import community
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
from numpy import polyfit, poly1d
sys.path.append("/home/yuanshuai20/paper/cluster_model/")
from data_process import load_graph,PortraitCommunity
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
            cluster_path = config.get(section, "cluster_path")
            # gae_model param config
        #数据处理
        nodes_path = pd.read_csv(nodes_path,index_col=0)
        graph,g_nodes = load_graph(matrix_file,nodes_path)

        resolution_range = [i / 10 for i in range(1, 51)]
        resolution_communities = []
        resolution_col = []
        resolution_mq = []
        for r in resolution_range:
            #louvain计算
            partition = community_louvain.best_partition(graph,resolution=r)
            #取出长度大于1的社区
            communities = {}
            for node, com in partition.items():
                if com in communities:
                    communities[com].append(node)
                else:
                    communities[com] = [node]
            communities = {k: v for k, v in communities.items() if len(v) > 1}
            communities_set = list(communities.values())
            modularity = community_louvain.modularity(partition, graph)
            resolution_mq.append(modularity)
            # 根据大小排序
            sorted_communities = sorted(communities_set, key=lambda b: -len(b))  # 按社区大小排序
            #写入文件
            count = 0
            with open(cluster_path, 'w') as f:
                for communitie in sorted_communities:
                    count += 1
                    # print("社区", count, " ",len(communitie) ,communitie)
                    PortraitCommunity(communitie,g_nodes,count,f)
            f.close()
            with open("/home/yuanshuai20/paper/互作对选择resolution.txt", "w") as f1:
                f1.write(str(resolution_communities) + '\n')
                f1.write(str(resolution_col) + '\n')
                f1.write(str(resolution_mq) + '\n')

            df = pd.read_table(cluster_path, header=None, names=['col1', 'col2', 'col3'], encoding='utf-8')
            df1 = df[['col1', 'col2']]
            df_new = pd.DataFrame(df1.to_numpy().reshape(-1, 1, order='F'), columns=['col1'])

            resolution_communities.append(len(communities))
            resolution_col.append(df_new['col1'].nunique())

        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[0].plot(resolution_range, resolution_communities)
        axs[0].set_xlabel('resolution')
        axs[0].set_ylabel('簇数')

        axs[1].plot(resolution_range, resolution_col)
        axs[1].set_xlabel('resolution')
        axs[1].set_ylabel('节点数')

        axs[2].plot(resolution_range, resolution_mq)
        axs[2].set_xlabel('resolution')
        axs[2].set_ylabel('模块度')
        plt.show()


if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(), 'config.cfg')
    train = Train()
    train.train_model(config_path)
