from collections import defaultdict
import numpy as np
import copy
import torch
from clients_and_server.cluster import Kmeans, K_means_cluster, Dbscan

class get_server(object):
    def __init__(self,model,device,clients,args):
        # self.model = copy.deepcopy(get_model(dataset=args.dataset).to(device))
        self.model = copy.deepcopy(model.to(device))
        self.device = device
        self.clients = clients
        self.args = args
        self.cluster_models = copy.deepcopy(model.to(device))

    # -----FedAvg aggregate all selected client models based on sample size-----
    def aggregate_model(self,select_clients,clients_sample_num):
        client_models = []
        sel_clients_sample_num = []
        p = []
        for i in select_clients:
            client_model = torch.load('./cache/client_model_{}.pt'.format(i))
            client_models.append(copy.deepcopy(client_model))
            sel_clients_sample_num.append(clients_sample_num[i])

        for i in range(len(select_clients)):
            p.append(sel_clients_sample_num[i] / sum(sel_clients_sample_num))
        print("Weights (based on sample size):" + str(p))

        # self.global_model = copy.deepcopy(self.model)
        param = copy.deepcopy(self.model.state_dict())

        for key in param.keys():
            param[key] = param[key] * 0
            for i in range(len(select_clients)):
                param[key] += p[i] * client_models[i][key]
        torch.save(param, './cache/global_model.pt')
        # self.global_model.load_state_dict(param)

    # -----Krum aggregate all selected client models-----
    def krum(self, select_clients,att_num):
        client_models = []
        for i in range(self.args.global_nums):
            if i in select_clients:
                client_model = torch.load('./cache/client_model_{}.pt'.format(i))
            else:
                client_model = torch.load('./cache/grad_model_{}.pt'.format(i))
            client_models.append(copy.deepcopy(client_model))

        param = copy.deepcopy(self.model.state_dict())
        distances = defaultdict(dict)
        non_malicious_count = int(len(select_clients) - att_num)
        num = 0
        for key in param.keys():
            if num == 0:
                for i in select_clients:
                    for j in select_clients:
                        distances[i][j] = distances[j][i] = np.linalg.norm(
                            client_models[i][key] - client_models[j][key])
                num = 1
            else:
                for i in select_clients:
                    for j in select_clients:
                        distances[j][i] += np.linalg.norm(client_models[i][key] - client_models[j][key])
                        distances[i][j] += distances[j][i]
        minimal_error = 1e20
        #print("user:")
        for user in distances.keys():
            #print(user,end="、")
            errors = sorted(distances[user].values())
            current_error = sum(errors[:non_malicious_count])
            if current_error < minimal_error:
                minimal_error = current_error
                minimal_error_index = user
        #print("distances:")
        #print(distances)
        print(minimal_error_index)
        torch.save(client_models[minimal_error_index], './cache/global_model.pt')

    #---------FLClusting-----------
    # Krum cluster aggregation
    def krum_in_cluster(self,select_clients,select_clients_id,att_num):
        client_models = []
        for i in range(self.args.global_nums):
            if i in select_clients_id:
                client_model = torch.load('./cache/client_model_{}.pt'.format(i))
                #print('client_model{}:{}'.format(i,client_model))
            else:
                client_model = torch.load('./cache/grad_model_{}.pt'.format(i))
            client_models.append(copy.deepcopy(client_model))#

        self.cluster_models = [copy.deepcopy(self.model.state_dict()) for _ in range(len(select_clients))]
        minimal_error_index=0
        for cluster_id,client_id in enumerate(select_clients):
            distances = defaultdict(dict)
            #print(att_num[cluster_id])
            non_malicious_count = int(len(client_id) - att_num[cluster_id])
            num = 0
            for key in self.cluster_models[cluster_id].keys():
                if num == 0:
                    for i in client_id:
                        for j in client_id:
                            distances[i][j] = distances[j][i] = np.linalg.norm(
                                client_models[i][key] - client_models[j][key])
                    num = 1
                else:
                    for i in client_id:
                        for j in client_id:
                            distances[j][i] += np.linalg.norm(client_models[i][key] - client_models[j][key])
                            distances[i][j] += distances[j][i]
            minimal_error = 1e20
            #print("user:")
            for user in distances.keys():
                #print(user,end="、")
                errors = sorted(distances[user].values())
                current_error = sum(errors[:non_malicious_count])
                if current_error < minimal_error:
                    minimal_error = current_error
                    minimal_error_index = user
            #print("distances:")
            #print(distances)
            #print(minimal_error_index)
            self.cluster_models[cluster_id] = client_models[minimal_error_index]
            torch.save(self.cluster_models[cluster_id], './cache/cluster_model_{}.pt'.format(cluster_id))

    # FedAvg aggregate all cluster models
    def aggregate_cluster_model(self):
        # self.global_model = copy.deepcopy(self.model)
        param = copy.deepcopy(self.model.state_dict())

        for key in param.keys():
            param[key] = param[key]*0
            for model in self.cluster_models:
                param[key] = param[key] + model[key]
            param[key] = torch.true_divide(param[key],len(self.cluster_models))
            # param[key] = param[key]/len(self.cluster_models)
        torch.save(param, './cache/global_model.pt')
        # self.global_model.load_state_dict(param)

    # a-FedAvg calculate weights
    def aggregate_cluster_model_basedon_q(self,local_acc):
        # self.global_model = copy.deepcopy(self.model)
        param = copy.deepcopy(self.model.state_dict())
        p = []
        for i in range(len(self.cluster_models)):
            if local_acc[i] == 0:
                local_acc[i] = 0.1
            local_acc[i] = local_acc[i] ** -0.5
        for i in range(len(self.cluster_models)):
            p.append(local_acc[i] / sum(local_acc))
        #print("Weighting (based on accuracy):"+ str(p))

        for key in param.keys():
            param[key] = param[key]*0
            for i in range(len(self.cluster_models)):
                param[key] += p[i] * self.cluster_models[i][key]
        torch.save(param, './cache/global_model.pt')
        # self.global_model.load_state_dict(param)

    def client_get_model(self,clients_id):
        pass

    #---------a-FedAvg-----------
    # FedAvg aggregate all models sent to clients (only used in the first round of a-FedAvg)
    def aggregate_q_model(self, select_clients):
        client_models = []
        for i in select_clients:
            client_model = torch.load('./cache/client_model_{}.pt'.format(i))
            client_models.append(copy.deepcopy(client_model))

        param = copy.deepcopy(self.model.state_dict())

        for key in param.keys():
            param[key] = param[key]*0
            for i in range(len(select_clients)):
                param[key] = param[key] + client_models[i][key]
            param[key] = torch.true_divide(param[key],len(select_clients))
            # param[key] = param[key]/len(self.cluster_models)
        torch.save(param, './cache/global_model.pt')
        # self.global_model.load_state_dict(param)

    # a-FedAvg calculates weights, aggregates models for all selected clients
    def aggregate_model_basedon_q(self, select_clients, local_acc):
        client_models = []
        p = []
        for i in select_clients:
            client_model = torch.load('./cache/client_model_{}.pt'.format(i))
            client_models.append(copy.deepcopy(client_model))

        for i in range(len(select_clients)):
            if local_acc[i] == 0:
                local_acc[i] = 0.1
            local_acc[i] = local_acc[i] ** -0.5
        for i in range(len(select_clients)):
            p.append(local_acc[i] / sum(local_acc))

        # self.global_model = copy.deepcopy(self.model)
        param = copy.deepcopy(self.model.state_dict())

        for key in param.keys():
            param[key] = param[key] * 0
            for i in range(len(select_clients)):
                param[key] += p[i] * client_models[i][key]
            # param[key] = torch.true_divide(param[key],len(select_clients))
            # param[key] = param[key]/len(self.cluster_models)
        torch.save(param, './cache/global_model.pt')
        # self.global_model.load_state_dict(param)
