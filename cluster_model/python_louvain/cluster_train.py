import os
import sys
from configparser import ConfigParser
from community import community_louvain
import pandas as pd
import torch
import community
sys.path.append("/home/yuanshuai20/paper/cluster_model/")
from data_process import load_graph,SingleCommunity
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

        #louvain计算
        partition = community_louvain.best_partition(graph,resolution=1)

        #取出长度大于1的社区
        communities = {}
        for node, com in partition.items():
            if com in communities:
                communities[com].append(node)
            else:
                communities[com] = [node]
        communities = {k: v for k, v in communities.items() if len(v) > 1}
        communities_set = list(communities.values())

        # 根据大小排序
        sorted_communities = sorted(communities_set, key=lambda b: -len(b))  # 按社区大小排序
        #写入文件
        count = 0
        with open(cluster_path, 'w') as f:
            for communitie in sorted_communities:
                count += 1
                print("社区", count, " ",len(communitie) ,communitie)
                SingleCommunity(communitie,graph,g_nodes,count,f)
        f.close()

        df = pd.read_table(cluster_path, header=None, names=['col1', 'col2', 'col3'], encoding='utf-8')
        print(str(len(communities)) + '/' + str(df.shape[0]))
        print(str(df['col1'].nunique()) + '/' + str(df['col2'].nunique()))


if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(), 'config.cfg')
    train = Train()
    train.train_model(config_path)
