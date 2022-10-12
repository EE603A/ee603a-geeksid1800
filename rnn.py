import torch
from torch.autograd import Variable
import torch.nn as nn
from torchsummary import summary


class RNNNetwork(nn.Module):
    def __init__(self):
        super(RNNNetwork, self).__init__()
        self.input_size = 128*500*2 #bands*frames*features
        self.hidden_size = 60
        self.num_layers = 2 #no. of hidden layers in LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        #self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(self.hidden_size/2), int(self.hidden_size/2))
        self.fc3 = nn.Linear(int(self.hidden_size/2), 10)
    
    def forward(self, x):
        x = torch.squeeze(x,dim=1)
        x = x.float()
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float())
        #out, _ = self.lstm(x, (h0,c0)) 
        #out = self.relu(self.fc1(out[:, -1, :]))
        #out = self.relu(self.fc2(out))
        #out = self.fc3(out) 
        #return out
        print(x.shape)
        return x

if __name__ == "__main__":
    rnn = RNNNetwork()
    summary(rnn, (1, 128, 500))