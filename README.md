# FLClustering

## 1.Introduction

	FLClustering is a fair and robust federated learning method for heterogeneous joint learning environments. 
    We integrate a Byzantine attack defence mechanism in the clustering after clustering participants with similar model parameters. 
    We introduce a dynamic adjustment strategy and a fair participant selection method in the aggregation process. 
    We employ an adaptive sample adjustment strategy to reduce the impact of sample size imbalance on clustering and further improve the effectiveness of joint learning.
    Experiments are performed on MNIST and Fashion-MNIST datasets.

## 2.File Structure
+ cache  : Storage of generated model files

+ clients_and_server

    + __init__.py
    + clients.py   :  Participants class
    + cluster.py   :  Clustering algorithms class
    + server.py    :  Data Centres class

+ data  :   Storing data sets

+ data_and_model   

    + __init__.py
    + datasets.py    :  Generate and divide datasets
    + models.py      :  Generate models

+ result :   Storage of experimental results

+ main：Main program

+ plot： Drawing diagrams


## 3.Detailed description

### 3.1 cache folder
      This folder holds the results of model training.
      Grad_model is the local pre-training model for clustering.
      Cient_model is the local training model.
      Global_model is the global aggregation model.
      Cluster_model is the cluster training model.

### 3.2 clients_and_server folder
    This folder contains three files, clients, cluster and server.

#### 3.2.1 clients file
    This file is used to define a user, a user's information: 
    self-label, cluster number, model, training test data, learning rate, optimiser, number of trainings, etc.
    Include functions: get_cluster_model, get_global_model, local_train, pre_train, nochange_pre_train, scaling_attack, label_attack, test_model, test_model_mid.
      get_cluster_modal：Get cluster models
      get_global_model：Get the global model
      local_train：Models are trained locally
      pre_train：Before model clustering, it is first necessary to pre-train to obtain a model that describes its own data
      nochange_pre_train: Pre-training for non-sample adjusted strategies      
      scaling_attack: Model Attack
      label_attack: Tab Flip Attack
      test_model_mid: Test model accuracy before aggregation for weight calculation
      test_model: Test model accuracy

#### 3.2.2 cluster file
      This file is to cluster  users using three clustering methods: k-means, hierarchical clustering and DBSCAN.

#### 3.2.3 server file
      This file is used to define data centres as well as models.
      aggregate_model：FedAvg aggregate all selected client models based on sample size
        krum: Krum aggregate all selected client model
        krum_in_cluster: Krum cluster aggregation
        aggregate_cluster_model：FedAvg aggregate all cluster models
        aggregate_cluster_model_basedon_q: Calculate weights
        aggregate_q_model: Aggregate all models sent to clients (only used in the first round of a-FedAvg)
        aggregate_model_basedon_q: Calculates weights, aggregates models for all selected clients
      
### 3.3 data folder
      This folder are used to store datasets, MNIST and FMNIST datasets can be used in this experiment.

### 3.4 data_and_model folder
      This folder holds 2 files, datasets and models.

#### 3.4.1 datasets file
      This file generates a class for generating data
      __load_dataset function is used to download the corresponding data in the data folder
      get_IID_data、get_nonIID_data and get_practical_nonIID_data is the 3 ways of generating data.
      The first one generates IID data, the second one generates non-IID data (each user's data contains two tags), and the third one is the more practical way.

#### 3.4.2 models file
      This file generates the initial model and returns a model via the get_model function.

### 3.5 main file
      The main function is the logic of the whole programme. 
      First client local pre-training, then clustering, followed by client local training, intra-cluster aggregation, and finally aggregation of all cluster models to form a global model.

### 3.6 plot file
      Drawing diagrams.

### 3.7 result folder
      Deposit results.








