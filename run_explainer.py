
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
from torch_geometric.explain import Explainer, GNNExplainer

import torch_geometric.transforms as T
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #not show plots interactively

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_list   = torch.load('train_custom_data.pt')

class Net(torch.nn.Module):
    def __init__(self, num_features, dim=16, num_classes=7):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, dim)
        self.conv2 = GCNConv(dim, num_classes)

    def forward(self, x, edge_index, data=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


dim = 16 #todo
n_classes = train_data_list[0].x.shape[1]
n_features = train_data_list[0].x.shape[1]
model = Net(num_features= n_features, dim=dim, num_classes=n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

PATH = "new_trained_model.pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()


explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)



node_ids_from_classes = {0:'0: kitchen',1:'1: fridge',2:'2: counter',3:'3: cabinet',4:'4: milk',5:'5:cereal', 6:'coffee',
                             7:'7: keys', 8:'8: cup', 9:'9: bowl', 10:'10: rack'}


graph_index =10
node_index = 2
x, edge_index = train_data_list[graph_index].x.to(device),train_data_list[graph_index].edge_index.to(device) #choose which graph to explain

explanation = explainer(x, edge_index, index=node_index)
print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=5)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path)
print(f"Subgraph visualization plot has been saved to '{path}'")

