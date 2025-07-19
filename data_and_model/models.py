import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------  MNIST  -------------------------
class MNIST_2NN(nn.Module):
    def __init__(self):
        super(MNIST_2NN,self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return F.log_softmax(tensor,dim=1)

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN,self).__init__()

        self.conv1 = nn.Sequential(        
            nn.Conv2d(1,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(     
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc1 = nn.Linear(7*7*32,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        output = out_conv2.view(-1,7*7*32)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return F.log_softmax(output,dim=1)

class MNIST_LR_Net(nn.Module):
    def __init__(self):
        super(MNIST_LR_Net, self).__init__()

        self.hidden1 = nn.Linear(28 * 28, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.hidden1(x), inplace=True) 
        x = F.relu(self.hidden2(x), inplace=True)
        x = self.out(x)

        return F.log_softmax(x, dim=1)

class MNIST_RNN(nn.Module):
    def __init__(self):
        super(MNIST_RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=28,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1,28,28)
        r_out, (h_n, h_c) = self.rnn(x, None)   

        out = self.out(r_out[:, -1, :])
        return out

#-----------------------get_model()--------------------------

def get_model(dataset,model_name="CNN",batch=None):

    if model_name == "2NN":
        model = MNIST_2NN()
    elif model_name == "CNN":
        model = MNIST_CNN()
    elif model_name == "MCLR":
        model = MNIST_LR_Net()
    elif model_name == "RNN":
        model = MNIST_RNN()
    else:
        model = "break"

    return model
