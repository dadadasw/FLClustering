import torch
import time
import random
random.seed(int(time.time())%100000)
from torch.utils.data import DataLoader, dataset, Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split
import sys

class download_data(object):
    def __init__(self, args):
        self.args = args
        self.data_name = args.dataset
        self.batch_size = args.batch_size
        self.get_data_way = args.get_data
        self.data_num = [[] for _ in range(args.k)]
        self.client_num = 0
        self.__load_dataset()# initial self.train_dataset and self.test_dataset
        self.__initial()
        self.index = 0
        self.train_sample_num = 0
        self.group = 0

    # initial self.train_dataset and self.test_dataset
    def __load_dataset(self,path = "/data"):
        # dataset path
        if self.data_name == 'MNIST':
            train_dataset = datasets.MNIST(path,
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

            test_dataset = datasets.MNIST(path,
                                          train=False,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))

        elif self.data_name == "FMNIST":
            train_dataset = datasets.FashionMNIST(path,
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.1307,), (0.3081,))
                                                  ]))
            test_dataset = datasets.FashionMNIST(path,
                                                 train=False,
                                                 download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))
                                                 ]))

        else:
            raise RuntimeError('the name inputed is wrong!')

        # self.train_dataset = list(train_dataset)
        # self.test_dataset = list(test_dataset)
        # print(self.train_dataset[0])
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # print(self.train_dataset[0])
        # sys.exit()

    def __initial(self):
        if self.get_data_way == "IID":  # IID, datasets are divided evenly
            num = int(len(self.train_dataset) / self.args.global_nums)
            train_split = [num for _ in range(self.args.global_nums)]
            train_split.append(len(self.train_dataset) - self.args.global_nums * num)

            num = int(len(self.test_dataset) / self.args.global_nums)
            test_split = [num for i in range(self.args.global_nums)]
            test_split.append(len(self.test_dataset) - self.args.global_nums * num)

            train_dataset = random_split(self.train_dataset, train_split)
            test_dataset = random_split(self.test_dataset, test_split)
            train_dataset = [DataLoader(train_dataset[i], batch_size=self.batch_size, shuffle=True,
                                        pin_memory=True) for i in range(self.args.global_nums)]
            test_dataset = [DataLoader(test_dataset[i], batch_size=self.batch_size, shuffle=True,
                                       pin_memory=True) for i in range(self.args.global_nums)]

            self.train_dataset = iter(train_dataset)
            self.test_dataset = iter(test_dataset)

        elif self.get_data_way == "nonIID":         # two classes per client
            self.ranked = []
            self.index_train = [[] for i in range(0, 10)]
            self.index_test = [[] for i in range(0, 10)]

            for i, data in enumerate(self.train_dataset): # index_train[0~9]
                self.index_train[data[1]].append(i)
            for i, data in enumerate(self.test_dataset):
                self.index_test[data[1]].append(i)

            # self.indexes = []

        else:
            if self.args.k == 3:
                self.index_train = [[], [], []]     # divide the datasets into three disjoint sets
                self.index_test = [[], [], []]
                for i, data in enumerate(self.train_dataset):
                    if data[1] < 3:
                        self.index_train[0].append(i)
                    elif data[1] > 5:
                        self.index_train[2].append(i)
                    else:
                        self.index_train[1].append(i)
                for i, data in enumerate(self.test_dataset):
                    if data[1] < 3:
                        self.index_test[0].append(i)
                    elif data[1] > 5:
                        self.index_test[2].append(i)
                    else:
                        self.index_test[1].append(i)

            else:
                self.index_train = [[], [], [], [], []]  # divide the datasets into five disjoint sets
                self.index_test = [[], [], [], [], []]
                for i, data in enumerate(self.train_dataset):
                    if data[1] < 2:
                        self.index_train[0].append(i)
                    elif data[1] < 4:
                        self.index_train[1].append(i)
                    elif data[1] < 6:
                        self.index_train[2].append(i)
                    elif data[1] < 8:
                        self.index_train[3].append(i)
                    else:
                        self.index_train[4].append(i)
                for i, data in enumerate(self.test_dataset):
                    if data[1] < 2:
                        self.index_test[0].append(i)
                    elif data[1] < 4:
                        self.index_test[1].append(i)
                    elif data[1] < 6:
                        self.index_test[2].append(i)
                    elif data[1] < 8:
                        self.index_test[3].append(i)
                    else:
                        self.index_test[4].append(i)


    # get IID datasets
    def get_IID_data(self):
        print("get_IID_data")

        # train_sample = random.sample(range(len(self.train_dataset)),train_num)
        # train_data = torch.utils.data.Subset(self.train_dataset,train_sample)
        # train_dataset = DataLoader(train_data,batch_size=self.batch_size, shuffle=True)
        #
        # test_sample = random.sample(range(len(self.test_dataset)),test_num)
        # test_data = torch.utils.data.Subset(self.test_dataset,test_sample)
        # test_dataset= DataLoader(test_data,batch_size=self.batch_size, shuffle=True)

        train_data = next(self.train_dataset)
        test_data = next(self.test_dataset)
        data = []
        data.append(train_data)
        data.append(test_data)
        return data


    def get_nonIID_data(self):
        print("get_nonIID_data")
        rank = random.sample(range(10),2)

        print(rank)

        train_index = []
        test_index = []
        for i in rank:
            train_index.extend(self.index_train[i])
            test_index.extend(self.index_test[i])

        train_index = random.sample(train_index, int(len(train_index) * 0.1))
        test_index = random.sample(test_index, int(len(test_index) * 0.2))
        sample_train = train_index
        self.train_sample_num = len(sample_train)
        #print(len(sample_train))
        sample_test = test_index

        dataset1 = torch.utils.data.Subset(self.train_dataset, sample_train)
        dataset2 = torch.utils.data.Subset(self.test_dataset, sample_test)
        train_dataset = DataLoader(dataset1, batch_size=self.batch_size, shuffle=True)
        test_dataset = DataLoader(dataset2, batch_size=self.batch_size, shuffle=True)

        data = []
        data.append(train_dataset)
        data.append(test_dataset)
        return data

    def get_practical_nonIID_data(self,id=None):
        if id == None:
            n = 3
        else:
            n = 5

        data_index = random.randint(0, n - 1)  # Gets the specified index dataset
        data_index = self.index % self.args.k
        self.index = self.index + random.randint(0, 100)
        #print(data_index)
        self.group = data_index
        # self.update_data(id=id,data=data_index)   # update
        train_data_num = 2000

        test_data_num = int(train_data_num * 0.2)  # 测试集大小

        sampling_train = random.sample(self.index_train[data_index], random.randint(train_data_num*0.1,train_data_num))  #200~2000 Obtain major component train_data
        self.train_sample_num = len(sampling_train)
        other_train = []
        for i in range(n):
            if i != data_index:
                other_train.extend(self.index_train[i])
        other_train = random.sample(other_train, int(train_data_num * 0.05))             # Obtain non-major component train_data
        sampling_train.extend(other_train)
        train_dataset = torch.utils.data.Subset(self.train_dataset, sampling_train)
        train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        sampling_test = random.sample(self.index_test[data_index], int(test_data_num))
        other_test = []
        for i in range(n):
            if i != data_index:
                other_test.extend(self.index_test[i])
        other_test = random.sample(other_test, int(test_data_num * 0.05))
        sampling_test.extend(other_test)
        test_dataset = torch.utils.data.Subset(self.test_dataset, sampling_test)
        test_dataset = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        data = []
        data.append(train_dataset)
        data.append(test_dataset)
        return data

    # update
    def update_data(self,id=None,data=None):
        if data == None:
            print("error...")
            sys.exit()
        
        if id == None:
            self.data_num[data].append(self.client_num)
            self.client_num += 1
        else:
            for data_index in self.data_num:
                if id in data_index:
                    data_index.remove(id)
            self.data_num[data].append(id)
            self.data_num[data].sort()
        for data_index in self.data_num:print(data_index)

    def get_data(self, get_data_way="train",id = None):
        if get_data_way == "test":
            data = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
            return data

        if self.get_data_way == "IID":
            data = self.get_IID_data()
        elif self.get_data_way == "nonIID":
            data = self.get_nonIID_data()
        else:
            data = self.get_practical_nonIID_data(id)

        return data


# Load the dataset and rewrite the DataSet class
class dataset(Dataset):
    def __init__(self, data_features, data_target):
        self.len = len(data_features)
        self.features = data_features
        self.target = data_target

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len

