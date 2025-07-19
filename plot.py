import math
import random

import numpy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
plt.rcParams['axes.unicode_minus']=False
losses = []
accuracyes = [[],[]]

def sava_data(acc,loss,var,data,dataset,attack_way,att_rate,aggregate_way,global_nums,change):
    dt = pd.DataFrame(acc)
    dt.to_csv("./result/acc_{}_{}_{}_{}_{}_{}_{}.csv".format(data,attack_way,att_rate,aggregate_way,dataset,global_nums,change), index=0)
    dt = pd.DataFrame(loss)
    dt.to_csv("./result/loss_{}_{}_{}_{}_{}_{}_{}.csv".format(data,attack_way,att_rate,aggregate_way,dataset,global_nums,change), index=0)
    dt = pd.DataFrame(var)
    dt.to_csv("./result/var_{}_{}_{}_{}_{}_{}_{}.csv".format(data,attack_way,att_rate,aggregate_way,dataset,global_nums,change), index=0)

def plot_acc(attack_way,rate,bigname,client_num):
    readData = pd.read_csv("./result/acc_practical_nonIID_"+attack_way+'_'+rate+"_FLClusting_MNIST_-1.csv", header=None)  # 读取csv数据
    y1 = readData.iloc[1:, 0].tolist()
    plt.xlabel("Global aggregation rounds")
    plt.ylabel("Average accuracy")
    x=np.arange(0,len(y1))
    plt.ylim(0,1.01)
    plt.plot(x, y1,label='FLClustering_mnist',color='darkorange')
    plt.savefig('./result/jpg/acc_practical_nonIID_'+attack_way+'_'+rate+'.jpg',dpi=200)
    plt.show()
    plt.clf()

def plot_loss(attack_way,rate,bigname,client_num):
    readData = pd.read_csv("./result/loss_practical_nonIID_"+attack_way+'_'+rate+"_FLClusting_MNIST_-1.csv", header=None)
    y1 = readData.iloc[1:, 0].tolist()
    plt.xlabel("Global aggregation rounds")
    plt.ylabel("Training loss")
    x=np.arange(0,len(y1))
    plt.plot(x, y1,label='FLClustering_mnist',color='darkorange')
    plt.savefig('./result/jpg/loss_practical_nonIID_'+attack_way+'_'+rate+'.jpg',dpi=200)
    plt.show()
    plt.clf()

def plot_var(attack_way,rate,client_num):
    readData = pd.read_csv("./result/var_practical_nonIID_"+attack_way+'_'+rate+"_FLClusting_MNIST_-1.csv", header=None)
    y1 = readData.iloc[1:, 0].tolist()
    y1 = numpy.sqrt(y1)
    plt.xlabel("Global aggregation rounds")
    plt.ylabel("Standard deviation")
    x=np.arange(10,len(y1)+10)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.plot(x, y1,label='FLCLustering',color='darkorange')
    plt.savefig('./result/jpg/var_practical_nonIID_'+attack_way+'_'+rate+'_mnist.jpg',dpi=200)
    plt.show()
    plt.clf()

if __name__ == "__main__":
    attack_way = "no"
    bigname = "no"
    rate = "0"
    dataset = "FMNIST"
    client_num = "100"
    plot_acc(attack_way,rate,bigname,client_num)
    plot_loss(attack_way,rate,bigname,client_num)
    plot_var(attack_way,rate,client_num)
