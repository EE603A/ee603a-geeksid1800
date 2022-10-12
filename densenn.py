import torch
from torch import nn
from torchsummary import summary

class DenseNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.pool = nn.AvgPool1d(kernel_size=8,stride=8)
        #self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(128*62, 128),
            nn.Dropout(p = 0.25),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(32,10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.squeeze(x,dim=1) #need a 2d or 3d tensor, but during runtime it is (batchsize,1,128,500) so we squeeze out dimension 1
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x) #logits
        #x = self.softmax(x) #predictions, if using CrossEntropy, softmax not needed
        return x
if __name__ == "__main__":
    ann = DenseNet()
    summary(ann, (1, 128, 500))