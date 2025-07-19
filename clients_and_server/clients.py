import numpy as np
from torch.utils.data import DataLoader

from data_and_model.models import get_model
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import copy
from data_and_model.datasets import dataset

class get_client(object):
    def __init__(self,id,model,device,dataset,args,train_sample_num,group):
        #client id
        self.id = id
        self.device = device
        self.args = args
        self.global_or_cluster_model = copy.deepcopy(model.to(device))
        self.train_data = dataset[0]
        self.test_data = dataset[1]
        self.train_sample_num = train_sample_num
        self.max_sample_num = 2000
        self.min_sample_num = 200
        self.group = group

        # self.model = copy.deepcopy(model.to(device))
        self.model = copy.deepcopy(get_model(dataset=args.dataset,model_name=args.model_name).to(device))
        self.model.load_state_dict(model.state_dict())
        self.optimizer = self.__initial_optimizer(args.optimizer,self.model,args.lr,args.momentum)
        self.scheduler_lr = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.gamma)

    def __initial_optimizer(self,optimizer,model,lr,momentum):             # initialize optimizer
        if optimizer == "SGD":
            opt = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
        else:
            opt = optim.Adam(params=model.parameters(),lr=lr)
        return opt

    # Pre-training before clustering to obtain a model that describes its own data
    def pre_train(self,epoch):
        #Initialised models
        grad_model = copy.deepcopy(self.global_or_cluster_model.to(self.device))
        grad_model.train()

        initial_model = copy.deepcopy(grad_model.state_dict())

        optimizer = optim.SGD(params=grad_model.parameters(), lr=0.05, momentum=0.9)
        for _ in range(epoch):

            data, target = next(iter(self.train_data))
            optimizer.zero_grad()
            # forward propagation
            output = grad_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            # backward propagation
            loss.backward()
            optimizer.step()

        grad_model = copy.deepcopy(grad_model.state_dict())

        for key in grad_model.keys():
            grad_model[key] = grad_model[key] - initial_model[key]

        torch.save(grad_model, './cache/grad_model_{}.pt'.format(self.id))

    # Pre-training for non-sample adjusted strategies
    def nochange_pre_train(self,epoch):

        grad_model = copy.deepcopy(self.global_or_cluster_model.to(self.device))

        grad_model.train()

        initial_model = copy.deepcopy(grad_model.state_dict())

        optimizer = optim.SGD(params=grad_model.parameters(), lr=0.05, momentum=0.9)
        for _ in range(epoch):
            for data, target in self.train_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                optimizer.zero_grad()

                output = grad_model(data)
                loss = nn.CrossEntropyLoss()(output, target)

                loss.backward()
                optimizer.step()

        grad_model = copy.deepcopy(grad_model.state_dict())

        for key in grad_model.keys():
            grad_model[key] = grad_model[key] - initial_model[key]

        torch.save(grad_model, './cache/grad_model_{}.pt'.format(self.id))

    # Model local training
    def local_train(self):

        self.model.train()

        for _ in range(self.args.local_epochs):
            for data, target in self.train_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                self.optimizer.step()
        
            self.scheduler_lr.step()

        torch.save(self.model.state_dict(), './cache/client_model_{}.pt'.format(self.id))

    # model attack
    def scaling_attack(self,scaling_factor):  # Scaling Attack
        client_model = copy.deepcopy(self.model.state_dict())
        for key in client_model.keys():
            client_model[key] = client_model[key] * scaling_factor
            self.model.state_dict()[key] = copy.deepcopy(client_model[key])

        torch.save(client_model, './cache/client_model_{}.pt'.format(self.id))

    #Tab Flip Attack
    def label_attack(self):
        data_ = []
        target_ = []
        for data, target in self.train_data:#per batch
            for idx in range(len(target)):
                if target[idx] == 0:
                    target[idx] = 9
                #if target[idx] == 9:
                    #target[idx] = 0
                #target[idx] = 9 - target[idx]
            target_.extend(target)
            data_.extend(data)
        train_dataset = DataLoader(dataset(data_,target_), batch_size=self.args.batch_size, shuffle=True)
        self.train_data = train_dataset

    #Test model accuracy before aggregation for weight calculation
    def test_model_mid(self):
        test_correct = 0
        global_model = copy.deepcopy(self.model)
        global_model.load_state_dict(torch.load('./cache/global_model.pt'))
        global_model.eval()
        with torch.no_grad():  
            for data, target in self.test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                output = global_model(data)

                test_correct += (torch.sum(torch.argmax(output,dim=1)==target)).item()

        return test_correct / len(self.test_data.dataset)

    #Test model accuracy
    def test_model(self):
        test_loss = 0
        test_correct = 0
        self.model.eval()

        with torch.no_grad():
            for data, target in self.test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                output = self.model(data)

                l = nn.CrossEntropyLoss()(output, target).item()
                test_loss += l
                test_correct += (torch.sum(torch.argmax(output,dim=1)==target)).item()

        return test_loss/len(self.test_data.dataset), test_correct / len(self.test_data.dataset)

    def get_global_model(self):
        self.model.load_state_dict(torch.load('./cache/global_model.pt'))

    #Get cluster models
    def get_cluster_model(self,clients_id):
        cluster = 0
        for cluster,clients in enumerate(clients_id):
            if self.id in clients:
                self.model.load_state_dict(torch.load('./cache/cluster_model_{}.pt'.format(cluster)))


