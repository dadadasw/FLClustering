import random

from data_and_model.models import get_model
from data_and_model.datasets import download_data
from clients_and_server.server import get_server
from clients_and_server.clients import get_client
from clients_and_server.cluster import Kmeans, K_means_cluster, Dbscan, Hac
from plot import sava_data
import numpy as np


import copy
import torch
import argparse
import time

def main(para):
    args = copy.deepcopy(para)

    print('Download and Initialize Dataset...')
    data_loader = download_data(args=args)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")

    model = get_model(dataset=args.dataset,model_name=args.model_name)
    clients = []
    clients_sample_num = []
    #Initialise clients
    for i in range(args.global_nums):
        #get data
        data = data_loader.get_data()
        #Initialise users
        user = get_client(id=i,model=copy.deepcopy(model),device=device,dataset=data,args=args,train_sample_num=data_loader.train_sample_num,group=data_loader.group)
        clients.append(user)
        print("Client:{} initialization is complete".format(i))
    for i in range(args.global_nums):
        clients_sample_num.append(clients[i].train_sample_num)
    print("sample size：",str(clients_sample_num))
    print("Grouping labels：",end="")
    for i in range(args.global_nums):
        print(clients[i].group,end=", ")
    #print('%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

    #Initialise server
    server = get_server(model=copy.deepcopy(model),device=device,clients=clients,args=args)

    losses = []
    accuracyes = []
    acc_var = []
    print("\n***************Start running**************")
    if args.aggregate_way == "FLClusting":
        print("Start pre-training......")
        if args.change == "no":
            for client in clients:
                client.nochange_pre_train(args.pre_epochs)
        else:
            for client in clients:
                client.pre_train(args.pre_epochs)
        #Clustering
        print("Start clustering......")
        #print(Dbscan(n_clients=args.global_nums))
        clients_id,result = Hac(n_clients=args.global_nums,k=args.k)
        # clients_id = Kmeans(n_clients=args.global_nums)
        print(clients_id)
        #data attack
        att_num = [0,0,0]
        attacker_id = []
        att_id_append = []
        order_other = []
        if args.attack_way == "data":
            for i, id in enumerate(clients_id):
                att_id = [i for i in id[0:round(len(id) * args.att_rate)]]
                attacker_id.extend(att_id)
                att_id_append.append(att_id)
                #print(att_id_append)
                for x in att_id_append[i]:
                    id.remove(x)
                order_other.append(id)
                #print(id)
            for id in attacker_id:
                clients[id].label_attack()
        for t_epoch in range(args.epoch):
            print("----------------global_epoch：{}---------------".format(t_epoch + 1))
            t_epoch_begin = time.time()
            if t_epoch > 0:
                local_acc = []
                for i, client_id in enumerate(clients_id):
                    order = np.random.permutation(client_id)
                    sid = [i for i in order[0:5]]
                    # print(str(sid))
                    acc_sum = 0
                    for id in sid:
                        acc = clients[id].test_model_mid()
                        # print(acc,end=" ")
                        acc_sum += acc
                    local_acc.append(acc_sum / 3)
                print(local_acc)
            # cluster_epoch
            for epoch in range(args.cluster_epoch):
                print("cluster_epoch：{}".format(epoch+1))
                # select clients
                select_clients_num = int(args.sel_client_nums / args.k)
                select_clients = []
                select_clients_id = []
                min = len(clients_id[0])
                #min_id = 0
                for i, client_id in enumerate(clients_id):
                    if len(clients_id[i]) < min:
                        min = len(clients_id[i])
                if min < select_clients_num :
                    select_clients_num = min
                if args.attack_way == "data":
                    for i, client_id in enumerate(clients_id):
                        random.shuffle(att_id_append[i])
                        random.shuffle(order_other[i])
                        clients_in_cluster = att_id_append[i][0:int(select_clients_num * args.att_rate)]
                        other_clients_in_cluster = [i for i in
                                                  order_other[i][0:int(select_clients_num * (1 - args.att_rate))]]
                        clients_in_cluster.extend(other_clients_in_cluster)
                        select_clients.append(clients_in_cluster)
                        select_clients_id.extend(clients_in_cluster)
                else:
                    for i, client_id in enumerate(clients_id):
                        order = np.random.permutation(client_id)
                        if i == args.k-1:
                            clients_in_cluster = [i for i in order[0:args.sel_client_nums-(select_clients_num*(args.k-1))]]
                        else:
                            clients_in_cluster = [i for i in order[0:select_clients_num]]
                        select_clients.append(clients_in_cluster)
                        select_clients_id.extend(clients_in_cluster)
                print("Random client selection within a cluster......" + str(select_clients))# 是id
                if args.attack_way == "data":
                    att_num = []
                    for i, id in enumerate(select_clients):
                        att_num.append(int(select_clients_num * args.att_rate))
                print("Local training......")
                for i in select_clients_id:
                    clients[i].local_train()
                if args.attack_way == "model":
                    attacker_id = []
                    if(t_epoch % 5 == 0):
                        att_num = []
                        for i,id in enumerate(select_clients):
                            att_id = [i for i in id[0:round(len(id) * args.att_rate)]]
                            att_num.append(round(len(id) * args.att_rate))
                            attacker_id.extend(att_id)
                        for id in attacker_id:
                            clients[id].scaling_attack(args.scaling_factor)
                if args.aggregate_cluster_way == "FedAvg":
                    print("FedAvg cluster model aggregation......")
                    server.aggregate_cluster(select_clients,select_clients_id)
                else:
                    print("Krum cluster model aggregation......")
                    server.krum_in_cluster(select_clients,select_clients_id,att_num)
                for client in clients:
                    client.get_cluster_model(clients_id)
            if t_epoch>0:
                print("Global aggregation......")
                server.aggregate_cluster_model_basedon_q(local_acc)
            else:
                server.aggregate_cluster_model()
            for client in clients:
                client.get_global_model()
            accuracyes.append(0)
            losses.append(0)
            acc_var.append(0)
            acc_arr = []
            for client in clients:
                loss,acc = client.test_model()
                acc_arr.append(acc)
                accuracyes[t_epoch] = accuracyes[t_epoch] + acc
                losses[t_epoch] = losses[t_epoch] + loss
            losses[t_epoch] = losses[t_epoch] / args.global_nums
            accuracyes[t_epoch] = accuracyes[t_epoch] / args.global_nums

            acc_var[t_epoch] = np.var(acc_arr)

            print("Average Accuracy：",accuracyes[t_epoch])
            print("Average loss：",losses[t_epoch])
            print("Global model variance：", acc_var[t_epoch])
    else:
        if args.attack_way == "data":
            order = np.random.permutation(args.global_nums)
            attacker_id = [i for i in order[0:int(args.global_nums * args.att_rate)]]
            order_other = order.tolist()
            for x in attacker_id:
                order_other.remove(x)

            for id in attacker_id:
                clients[id].label_attack()
        for epoch in range(args.epoch):
            print("----------------global_epoch：{}---------------".format(epoch + 1))
            order = np.random.permutation(args.global_nums)
            if args.attack_way == "data":
                random.shuffle(attacker_id)
                random.shuffle(order_other)
                clients_in_epoch = attacker_id[0:int(args.sel_client_nums * args.att_rate)]
                print("被选择的恶意客户端："+ str(clients_in_epoch))
                other_clients_in_epoch = [i for i in order_other[0:int(args.sel_client_nums * (1-args.att_rate))]]
                clients_in_epoch.extend(other_clients_in_epoch)
                print("随机选择客户端......" + str(clients_in_epoch))  # 是id
            else:
                clients_in_epoch = [i for i in order[0:args.sel_client_nums]]
            if args.aggregate_way == "α-FedAvg":
                if epoch > 0:
                    local_acc =[]
                    for id in clients_in_epoch:
                        acc = clients[id].test_model_mid()
                        local_acc.append(acc)
                    print(local_acc)
            print("Local training......")
            for i in clients_in_epoch:
                clients[i].local_train()
            if args.attack_way == "model":
                if (epoch % 5 == 0):
                    order2 = np.random.permutation(clients_in_epoch)
                    attacker_id = [i for i in order2[0:int(args.sel_client_nums * args.att_rate)]]
                    for id in attacker_id:
                        clients[id].scaling_attack(args.scaling_factor)
            if args.aggregate_way == "FedAvg":
                print("FedAvg......")
                server.aggregate_model(clients_in_epoch,clients_sample_num)
            if args.aggregate_way == "Krum":
                print("Krum......")
                server.krum(clients_in_epoch,int(args.sel_client_nums * args.att_rate))
            if args.aggregate_way == "α-FedAvg":
                if epoch > 0:
                    print("Global aggregation......")
                    server.aggregate_model_basedon_q(clients_in_epoch,local_acc)
                else:
                    server.aggregate_q_model(clients_in_epoch)
            for client in clients:
                client.get_global_model()
            accuracyes.append(0)
            losses.append(0)
            acc_var.append(0)
            acc_arr = []
            for client in clients:
                loss, acc = client.test_model()
                #print(acc, end=",")
                acc_arr.append(acc)
                accuracyes[epoch] = accuracyes[epoch] + acc
                losses[epoch] = losses[epoch] + loss
            losses[epoch] = losses[epoch] / args.global_nums
            accuracyes[epoch] = accuracyes[epoch] / args.global_nums
            acc_var[epoch] = np.var(acc_arr)
            print("Average accuracy：", accuracyes[epoch])
            print("Average loss：", losses[epoch])
            print("Global model variance：", acc_var[epoch])

    sava_data(accuracyes, losses,acc_var,args.get_data, args.dataset, args.attack_way, args.att_rate,args.aggregate_way,args.global_nums,args.change)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate_way", type=str, default="FLClusting", choices=["FedAvg","Krum","FLClusting","α-FedAvg"])
    parser.add_argument("--attack_way", type=str, default="model", choices=["no","model","data"])
    parser.add_argument("--att_rate", type=float, default=0.3,help="Attacker rate of sel_client_nums")
    parser.add_argument("--epoch", type=int, default=100, help="communicate epoch")#100
    parser.add_argument("--global_nums", type=int, default=10, help="Number of all Users")#100
    parser.add_argument("--sel_client_nums", type=int, default=5,help="select client nums")#
    parser.add_argument("--dataset", type=str, default="FMNIST", choices=["MNIST","FMNIST"])

    parser.add_argument("--get_data", type=str, default="practical_nonIID", choices=["IID","nonIID","practical_nonIID"])
    parser.add_argument("--model_name", type=str, default="CNN", choices=["CNN","MCLR","RNN","ResNet18"])
    parser.add_argument("--scaling_factor", type=int, default=-1,help="Attacker scaling factor")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01,help="Local learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, choices=[1, 0.98, 0.99])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--cluster_epoch", type=int, default=1,help="cluster epoch")#1
    parser.add_argument("--local_epochs", type=int, default=3)#3
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to run,-1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--change", type=str, default="yes",choices=["yes","no"],help="Sample adjustment or not")
    parser.add_argument("--k", type=int, default=3, help="Number of all clusters")
    parser.add_argument("--pre_epochs", type=int, default=5, help="Number of pre epochs")
    parser.add_argument("--aggregate_cluster_way", type=str, default="trum", choices=["trum","FedAvg"])
    args = parser.parse_args()

    print("=" * 80)  
    print("Summary of training process:")
    print("Dataset: {}".format(args.dataset))        # default="MNIST", choices=["MNIST", "CIFAR10", "EMNIST"]
    print("Get data way: {}".format(args.get_data))  # default="IID", choices=["IID","nonIID","practical_nonIID"]
    print("Attack way: {}".format(args.attack_way))
    print("Aggregate way: {}".format(args.aggregate_way))
    print("Model_name: {}".format(args.model_name))
    print("Batch size: {}".format(args.batch_size))  # default=20
    print("Learning rate: {}".format(args.lr))  # default=0.01, help="Local learning rate"
    print("gamma: {}".format(args.gamma))
    print("Momentum: {}".format(args.momentum))
    print("Optimizer: {}".format(args.optimizer))  # default="SGD"
    print("pre_epoch: {}".format(args.pre_epochs))
    print("cluster_epoch: {}".format(args.cluster_epoch))
    print("epoch: {}".format(args.epoch))
    print("local_epoch: {}".format(args.local_epochs))
    print("Number of clusters: {}".format(args.k))
    print("Select users: {}".format(args.sel_client_nums))
    print("All users: {}".format(args.global_nums))     # default=100, help="Number of all Users"
    print("=" * 80)

    main(args)
