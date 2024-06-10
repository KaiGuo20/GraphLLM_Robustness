import argparse
import torch
from attacks.pyg_defence import GCN, SAGE
from deeprobust.graph.utils import *
from copy import deepcopy
import json

from attacks.models import *
import attacks.models as models
from attacks.node_prbcd_v1 import *
from raw_data.data import *
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.ticker as ticker

import ipdb




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora")
parser.add_argument('--data_type', type=str, default="fixed")
parser.add_argument('--model', type=str, default='sbert')
parser.add_argument('--victim', type=str, default='mlp')
parser.add_argument('--sample_choice', type=str, default="random")
parser.add_argument('--edge_ptb', type=float, default=0)
parser.add_argument('--node_ptb', type=float, default=0.05)
args = parser.parse_args()


# save_path = 'eval_results'
save_path = 'used_eval'

# tokenizer = AutoTokenizer.from_pretrained(args.model)

dataset = torch.load(f'processed_data/ptb_datasets/{args.victim}_{args.dataset}_{args.data_type}_{args.model}_{args.node_ptb}.pt', map_location = 'cpu')

all_sa_texts = []
all_fa_texts = []
all_sa_feats = torch.FloatTensor([])
all_fa_feats = torch.FloatTensor([])

all_feats = torch.FloatTensor([])
all_labels = torch.LongTensor([])


total_success_idx = []
total_failed_idx = []
total_mod_idx = []

for split in range(5):
    mod_nodes = dataset.mod_nodes[split]
    # ptb_texts = dataset.ptb_texts[split]
    ptb_texts = dataset.raw_texts
    # perturbed
    # ptb_features = dataset.ptb_features[split]
    # original 
    ptb_features = torch.FloatTensor(dataset.x)

    mod_result = dataset.mod_result[split]
    assert list(mod_result[0].keys()) == list(mod_result[1].keys())
    # ipdb.set_trace()
    clean_label = torch.Tensor(list(mod_result[1].values())).bool()
    ptb_clean = torch.Tensor(list(mod_result[0].values())).bool()

    correct_idx = torch.where(clean_label==True)[0]
    attacked_idx = torch.where(ptb_clean==False)[0]
    success_idx = []
    failed_idx = []

    for idx in correct_idx:
        node_idx = list(mod_result[0].keys())[idx]
        if idx in attacked_idx:
            success_idx.append(node_idx)
        else:
            failed_idx.append(node_idx)

    total_success_idx += success_idx 
    total_failed_idx += failed_idx 
    total_mod_idx += mod_nodes.tolist()
    
    mod_texts = [ptb_texts[i] for i in mod_nodes]
    mod_feats = torch.stack([ptb_features[i] for i in mod_nodes])
    mod_labels = torch.stack([dataset.y[i] for i in mod_nodes])

    all_feats = torch.cat([all_feats, mod_feats])
    all_labels =torch.cat([all_labels, mod_labels])

    sa_texts = [ptb_texts[i] for i in success_idx]
    fa_texts = [ptb_texts[i] for i in failed_idx]
    sa_feats = torch.stack([ptb_features[i] for i in success_idx])
    fa_feats = torch.stack([ptb_features[i] for i in failed_idx])

    all_sa_texts += sa_texts
    all_fa_texts += fa_texts
    # ipdb.set_trace()
    all_sa_feats = torch.cat([all_sa_feats, sa_feats])
    all_fa_feats = torch.cat([all_fa_feats, fa_feats])


count_vectorizer = CountVectorizer()
# 构造词频矩阵
count_vectorizer = count_vectorizer.fit(all_fa_texts+all_sa_texts)
tokenizer = count_vectorizer.build_tokenizer()
# 获取特征词
feature_names = count_vectorizer.get_feature_names_out()
# 词频矩阵
sa_x = count_vectorizer.transform(all_sa_texts).toarray()
fa_x = count_vectorizer.transform(all_fa_texts).toarray()


#---------------words count
sa_avg_words = sa_x.astype(bool).sum(1).mean()
fa_avg_words = fa_x.astype(bool).sum(1).mean()
print(f"sa_avg_len: {sa_avg_words}")
print(f"fa_avg_len: {fa_avg_words}")

#---------------text length
sa_avg_len = sa_x.sum(1).mean()
fa_avg_len = fa_x.sum(1).mean()
print(f"sa_avg_len: {sa_avg_len}")
print(f"fa_avg_len: {fa_avg_len}")

#---------------entropy analysis
def avg_H(X):
    h = []
    for idx in range(len(X)):
        txt = X[idx]
        txt = txt/txt.sum()
        h.append(sum([-p*np.log2(p) for p in txt if p>0]))
    return np.mean(h)
from copy import deepcopy
sa_en = avg_H(deepcopy(sa_x))
fa_en = avg_H(deepcopy(fa_x))

print(f"sa_en: {sa_en}")
print(f"fa_en: {fa_en}")
    
#---------------LDA analysis
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pandas as pd

lda = LDA(n_components=3, random_state=0)

# label1 = np.zeros(len(all_fa_texts), dtype = int)
# label2 = np.ones(len(all_sa_texts), dtype = int)
# label = np.concatenate([label1, label2])
X = np.concatenate([np.array(fa_x), np.array(sa_x)])
lda_result = lda.fit_transform(X)
fa_lda = lda_result[:fa_x.shape[0],:].mean(0)
sa_lda = lda_result[:sa_x.shape[0],:].mean(0)

plt.figure()

# 将向量转换为长格式的 DataFrame
data = pd.DataFrame({
    'Value': list(sa_lda) + list(fa_lda),
    'types': ['success'] * 3 + ['failed'] * 3,
    'Group': ['Group 1', 'Group 2', 'Group 3', ] * 2 # 'Group 4', 'Group 5'
})

# sns.set_palette('colorblind')

# 使用 Seaborn 绘制柱状图
plt.figure( dpi=250) #figsize=(10,5), 
sns.barplot(x='Group', y='Value', hue='types', palette=['#ECA47C', '#759DDB'], data=data, width=0.5)
plt.legend(prop={'size':16})
plt.title(args.dataset, fontsize=20)
plt.xlabel("Theme", fontsize = 16)
plt.ylabel("Value", fontsize = 16)
plt.tick_params(labelsize='large')

plt.savefig(f"{save_path}/lda_{args.model}_{args.data_type}_{args.dataset}_{args.victim}.jpg")



# #---------------features embedding analysis
# from sklearn.manifold import TSNE 

# # label_fa = np.zeros(len(all_fa_feats), dtype = int)
# # label_sa = np.ones(len(all_sa_feats), dtype = int)
# # label = np.concatenate([label_fa, label_sa]).astype(int)

# label_fa = dataset.y[total_failed_idx]
# label_sa = dataset.y[total_success_idx]
# label = np.concatenate([label_fa, label_sa]).astype(int)

# # ipdb.set_trace()
# feats = np.array(torch.cat([all_fa_feats, all_sa_feats]))

# tsne = TSNE(n_components=2) 
# X_tsne = tsne.fit_transform(feats) 
# # ipdb.set_trace()
# X_tsne_data = np.vstack((X_tsne.T, label)).T 
# df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class']) 
# df_tsne.head()

# plt.figure(figsize=(8, 8)) 
# sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2') 
# plt.savefig(f'{save_path}/tsne_{args.model}_{args.data_type}_{args.dataset}_{args.victim}.jpg')

# #--------------------DBI
# from sklearn.metrics import davies_bouldin_score
# import numpy as np

# # 计算 DBI
# # ipdb.set_trace()
# dbi = davies_bouldin_score(all_feats.numpy(), all_labels.numpy())
# print("DBI:", dbi)

# #----------------save
# result = {'successfully attacked: Entropy': sa_en, 
#           'failed attacked: Entropy': fa_en, 
#           'successfully attacked: text length': sa_avg_len,
#           'failed attacked: text length': fa_avg_len,
#           'successfully attacked: number of words': sa_avg_words,
#           'failed attacked: number of words': fa_avg_words,
#           'DBI': dbi
#           }

# with open(f"{save_path}/{args.model}_{args.data_type}_{args.dataset}_{args.victim}.json","w") as f:
#     json.dump(result,f)


#---------------degree-asr
# total_degrees = []
# for node in total_mod_idx:
#     degree = len(torch.where(dataset.edge_index[0] == node)[0])
#     total_degrees.append(degree)
success_degree = []
for node in total_success_idx:
    degree = len(torch.where(dataset.edge_index[0] == node)[0])
    success_degree.append(degree)
failed_degree = []
for node in total_failed_idx:
    degree = len(torch.where(dataset.edge_index[0] == node)[0])
    failed_degree.append(degree)

# result = {}
# for i in set(total_degrees):
#     if target_degree.count(i)!=0 and total_degrees.count(i)!=0:
#         result[i] = (float(target_degree.count(i))/sum(target_degree)) * (float(sum(total_degrees))/total_degrees.count(i))

# x = list(result.keys())
# y = list(result.values())

# ipdb.set_trace()

plt.rcParams['font.sans-serif']=['SimSun'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

plt.figure(figsize=(8,6), dpi=250)
plt.figure()
# sns.kdeplot(success_degree, label='successfully attacked', shade=True)
# sns.kdeplot(failed_degree, label='failed attacked', shade=True)

sns.kdeplot(success_degree, label=u'攻击成功', shade=True)
sns.kdeplot(failed_degree, label=u'攻击失败', shade=True)
plt.legend(prop={'size':18})
# plt.title(args.dataset, fontsize=20)
# plt.xlabel("Degree", fontsize = 18)
# plt.ylabel("Density", fontsize = 18)
plt.xlabel(u"节点度", fontsize = 18)
plt.ylabel(u"概率密度", fontsize = 18)
plt.tick_params(labelsize=20)
plt.ticklabel_format(style='sci', scilimits=(0,10))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 1))
plt.gca().yaxis.get_offset_text().set_fontsize(18)
plt.gca().xaxis.get_offset_text().set_fontsize(18)

plt.subplots_adjust(bottom=0.15)

plt.savefig(f"{save_path}/degree_{args.model}_{args.data_type}_{args.dataset}_{args.victim}.pdf")


# ---------------centrality analysis
import networkx as nx

G = nx.Graph()

G.add_nodes_from(list(range(len(dataset.x))))

edge_index = dataset.edge_index
edge_index = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.shape[1])]
G.add_edges_from(edge_index)

# 计算介数中心性

pagerank = nx.pagerank(G)
eigenvector_centrality = nx.eigenvector_centrality_numpy(G)

sa_rank = [pagerank[idx] for idx in total_success_idx]
fa_rank = [pagerank[idx] for idx in total_failed_idx]

sa_eigen = [eigenvector_centrality[idx] for idx in total_success_idx]
fa_eigen = [eigenvector_centrality[idx] for idx in total_failed_idx]

# ipdb.set_trace()

# draw pagerank_asr
# plt.figure(figsize=(8,6), dpi=250) #figsize=(10,5), 
plt.figure()
# sns.kdeplot(sa_rank, label='successfully attacked', shade=True)
# sns.kdeplot(fa_rank, label='failed attacked', shade=True)
sns.kdeplot(sa_rank, label=u'攻击成功', shade=True)
sns.kdeplot(fa_rank, label=u'攻击失败', shade=True)

plt.legend(prop={'size':18})
# plt.title(args.dataset, fontsize=20)
# plt.xlabel("PageRank", fontsize = 18)
# plt.ylabel("Density", fontsize = 18)
plt.xlabel(u"PageRank值", fontsize = 18)
plt.ylabel(u"概率密度", fontsize = 18)
plt.tick_params(labelsize=20)
plt.ticklabel_format(style='sci', scilimits=(0,10))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 1))
plt.gca().yaxis.get_offset_text().set_fontsize(18)
plt.gca().xaxis.get_offset_text().set_fontsize(18)

plt.subplots_adjust(bottom=0.15)
# plt.ylabel("probability")
plt.savefig(f"{save_path}/pagerank_{args.model}_{args.data_type}_{args.dataset}_{args.victim}.pdf")

# draw eigen_asr
plt.figure(figsize=(8,6), dpi=250) #figsize=(10,5), 
plt.figure()
# sns.kdeplot(sa_eigen, label='successfully attacked', shade=True)
# sns.kdeplot(fa_eigen, label='failed attacked', shade=True)
sns.kdeplot(sa_eigen, label=u'攻击成功', shade=True)
sns.kdeplot(fa_eigen, label=u'攻击失败', shade=True)
plt.legend(prop={'size':18})
# plt.title(args.dataset, fontsize=20)
# plt.xlabel("Eigenvector Centrality", fontsize = 18)
# plt.ylabel("Density", fontsize = 18)
plt.xlabel(u"特征向量中心性", fontsize = 18)
plt.ylabel(u"概率密度", fontsize = 18)
plt.tick_params(labelsize=20)
plt.ticklabel_format(style='sci', scilimits=(0,10))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 1))
plt.gca().yaxis.get_offset_text().set_fontsize(18)
plt.gca().xaxis.get_offset_text().set_fontsize(18)

plt.subplots_adjust(bottom=0.15)

# plt.ylabel("probability")
plt.savefig(f"{save_path}/eigen_{args.model}_{args.data_type}_{args.dataset}_{args.victim}.pdf")


    

