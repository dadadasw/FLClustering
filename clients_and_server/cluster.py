import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import time
import random
import torch
import copy
import matplotlib.pyplot as plt

#k-means
def K_means_cluster(n_clients,k_means):
    clients = load_clients(n_clients=n_clients)
    a = torch.transpose(torch.tensor(clients),0,1)     
    u,s,v = torch.svd_lowrank(a,q=n_clients)

    a = s[0:30]
    plt.pie(a,autopct='%1.1f%%')
    plt.savefig("a.jpg")
    plt.show()


    initial = random.sample(range(0, n_clients), k_means)  
    init_model = [clients[i] for i in initial]
    indexes2 = [[] for i in range(k_means)] 
    print("init_model:",initial)
    print("shape:",np.array(clients).shape)
    num = 1

    while True:
        clusters = [[] for i in range(k_means)]   
        indexes = [[] for i in range(k_means)]      

        for i in range(n_clients):
      
            distance = []
            for j in range(k_means):
                a = np.sqrt(np.sum(np.square(np.array(clients[i])-np.array(init_model[j]))))
                distance.append(a)
            id = distance.index(min(distance))
            clusters[id].append(clients[i])
            indexes[id].append(i)


        print(np.array(clusters).shape)
        for i in range(k_means):
            print(i)
            a = np.array(clusters[i][0])*0.0
            for j in clusters[i]:
                a = np.array(a) + np.array(j)
            a = np.array(a) / len(clusters[i])
            init_model[i] = a.tolist()

        print(num, indexes, indexes2,'\n')
        if (np.array(indexes).shape == np.array(indexes2).shape) and (np.array(indexes) == np.array(indexes2)).all():
            if num % 100 == 0:
                print("无法收敛，返回")
                return indexes
                
            break
        else:
            num = num + 1
            indexes2 = copy.deepcopy(indexes)

    return indexes

#K-Means Library Function
def Kmeans(n_clients):
    clients = load_clients(n_clients=n_clients)
    u,s,v = torch.svd_lowrank(torch.tensor(clients),q=n_clients)

    for k in range(len(s)):             
        if torch.sum(s[0:k])/torch.sum(s) >= 0.5:
            break
    result = KMeans(k,max_iter=100).fit(clients).labels_    # kmeans

    cluster = [[] for _ in range(k)]
    for i,index in enumerate(result):
        cluster[index].append(i)

    return cluster

#DBSCAN
def Dbscan(n_clients):
    clients = load_clients(n_clients=n_clients)
    u, s, v = torch.svd_lowrank(torch.tensor(clients), q=n_clients)
    for k in range(len(s)):
        if torch.sum(s[0:k])/torch.sum(s) >= 0.8:
            break
    result = DBSCAN(eps=1, min_samples=1,metric='euclidean', algorithm='auto').fit(clients).labels_
    print("DBSCAN lables：")
    print(result)
    cluster = [[] for _ in range(len(result))]
    for i,index in enumerate(result):
        cluster[index].append(i)

    return result

#HAC
def Hac(n_clients,k):
    clients = load_clients(n_clients=n_clients)
    model = AgglomerativeClustering(distance_threshold=None,
                                    linkage="ward",
                                    n_clusters=k,
                                    affinity='euclidean',
                                    compute_distances=True).fit(clients)
    result = model.labels_
    cluster_score_si = metrics.silhouette_score(clients, model.labels_)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("hierarchical clustering dendrogram")
    # Tree diagram
    plot_dendrogram(model, truncate_mode="level")
    plt.xlabel("Client id/Number of data in cluster")
    plt.show()
    cluster = [[] for _ in range(k)]
    for i, index in enumerate(result):
        cluster[index].append(i)
    return cluster,result


# Load all user models from . /cache/grad_model_{}.pt to load all user models
def load_clients(n_clients: object) -> object:
    model_states = [[] for i in range(n_clients)]      
    for i in range(n_clients):
        grad_model = torch.load('./cache/grad_model_{}.pt'.format(i))       
        
        for name in grad_model.keys():
            a = grad_model[name].view(-1).tolist()
            #print(name)
            model_states[i].extend(a)
            
    return model_states

# Define a function that creates a tree diagram
def plot_dendrogram(model, **kwargs):

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Tree diagram
    dendrogram(linkage_matrix, **kwargs)