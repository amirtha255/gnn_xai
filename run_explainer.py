
import torch
import torch_geometric
import torch.nn.functional as F
import networkx as nx
import numpy as np
import argparse
import yaml
import os

from torch_geometric.nn import SplineConv
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Set2Set
from torch_geometric.explain import Explainer, GNNExplainer , PGExplainer, CaptumExplainer
from torch_geometric.visualization import visualize_graph

import random
random.seed(23435)
from numpy import random as nrandom
nrandom.seed(23435)

import torch_geometric.transforms as T
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #not show plots interactively


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
def simple_model_explanations(cfg):
    train_data_list   = torch.load(cfg['data_loader_path'])
    dim = 16 #todo
    n_classes = train_data_list[0].x.shape[1]
    n_features = train_data_list[0].x.shape[1]
    model = Net(num_features= n_features, dim=dim, num_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

    checkpoint = torch.load(cfg['model_ckpt'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    model_config=dict(
            mode=cfg['mode'],
            task_level=cfg['task_level'],
            return_type=cfg['return_type'],
    )
    
    algorithm=None
    if cfg['algorithm'] == 'GNNExplainer':
        algorithm = GNNExplainer(epochs=cfg['explanation_epochs'])
    elif cfg['algorithm'] == 'PGExplainer':
        algorithm = PGExplainer(epochs=cfg['explanation_epochs'])
    elif cfg['algorithm'] == 'CaptumExplainer':
        algorithm = CaptumExplainer(attribution_method='Integrated_Gradients')
    else:
        print('not a valid explanation algorithm')
        exit(0)

    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type=cfg['explanation_type'],
        node_mask_type=cfg['node_mask_type'],
        edge_mask_type=cfg['edge_mask_type'],
        model_config=model_config,
    )

    node_ids_from_classes = {0:'0: kitchen',1:'1: fridge',2:'2: counter',3:'3: cabinet',4:'4: milk',5:'5:cereal', 6:'coffee',
                                7:'7: keys', 8:'8: cup', 9:'9: bowl', 10:'10: rack'}

    graph_index = cfg['graph_index']
    node_index = cfg['node_index']
    #print('graph and node index',graph_index,node_index)
    target = (train_data_list[graph_index].y).long().to(device) #todo choice in test list
    x, edge_index = train_data_list[graph_index].x.to(device),train_data_list[graph_index].edge_index.to(device) #choose which graph to explain

    explanation = explainer(x, edge_index, index=node_index, target=target)
    #print(f'Generated explanations in {explanation.available_explanations}')
    #print('Edge mask sum ',explanation.edge_mask.cpu().numpy())

    if not os.path.exists(cfg['output_dir']):
        os.makedirs(cfg['output_dir'])

    path = os.path.join(cfg['output_dir'],'feature_importance_g{}_n{}_{}.png'.format(graph_index,node_index,cfg['name']))
    explanation.visualize_feature_importance(path, top_k=5)
    #print(f"Feature importance plot has been saved to '{path}'")

    path = os.path.join(cfg['output_dir'],'subgraph_g{}_n{}_{}.pdf'.format(graph_index,node_index,cfg['name']))
    explanation.visualize_graph(path)
    #print(f"Subgraph visualization plot has been saved to '{path}'")

    path = os.path.join(cfg['output_dir'],'orig_graph_g{}_n{}_{}.pdf'.format(graph_index,node_index,cfg['name']))
    visualize_graph(edge_index,None,path,None)
    
    top_feature_imp = explanation.node_mask.sum(dim=0).cpu().numpy()
    top_5_features = (-top_feature_imp).argsort()[:5]
    
    print('Top 5 features are',top_5_features, [top_feature_imp[ind] for ind in top_5_features] )
    print('Explanantion edge mask is ',explanation.edge_mask)

    return top_5_features, [top_feature_imp[ind] for ind in top_5_features]

if __name__ == '__main__':
    
    with open('xai_config/default.yaml') as f:
        cfg = yaml.safe_load(f)

    simple_model_explanations(cfg)
    
    #options to override cfg
    
    
    