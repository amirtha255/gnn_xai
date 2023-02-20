
import torch
import torch_geometric
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.nn import SplineConv
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Set2Set

from time import sleep
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #not show plots interactively


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
A NN to predict the next state of nodes
"""
class Net(torch.nn.Module):
    def __init__(self, num_features, dim=16, num_classes=7,**kwargs):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, dim)
        self.conv2 = GCNConv(dim, num_classes)
        self.model_cfg = kwargs

    def forward(self, x, edge_index, **kwargs):       
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(loader, model): 
    model.train() 
    for data in loader:
        optimizer.zero_grad() 
        out = model(data.x.to(device), data.edge_index.to(device))
        data.y = data.y.type(torch.LongTensor)
        loss = criterion(out[data.train_mask].to(device), data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        
def test(loader, model):
    model.eval()
    total_nodes = 0
    correct = 0
    for data in loader:
        out = model(data.x.to(device), data.edge_index.to(device)) # drop edges
        pred = out.argmax(dim=1)   
        correct += int((pred.to(device) == data.y.to(device)).sum())
        total_nodes += data.num_nodes
    return correct/total_nodes

if __name__ == '__main__':
        
    train_data_loader = DataLoader(torch.load('train_custom_data.pt'))
    train_data_list   = torch.load('train_custom_data.pt')
    test_data_loader = DataLoader(torch.load('test_custom_data.pt'))

    train_epoch = 100
    train_acc = []
    test_acc = []
    dim = 16
    n_classes = train_data_list[0].x.shape[1]
    n_features = train_data_list[0].x.shape[1]
    model = Net(num_features= n_features, dim=dim, num_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    
    for epoch in range(1, train_epoch+1):
        train(train_data_loader, model)
        train_acc.append(test(train_data_loader, model))
        test_acc.append(test(test_data_loader, model))

    
    

    PATH = "trained_simple_model.pt"    
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': test_acc,
            }, PATH)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(np.arange(train_epoch), np.array(train_acc) ) 
    ax1.set_title(label = 'Accuracy - training')
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(np.arange(train_epoch), np.array(test_acc))
    ax2.set_title(label = 'Test')
    
    fig.savefig('model_training_losses.png')