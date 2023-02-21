"""
to run spatio model with new dataset and gnn explainer
"""
import torch
import torch_geometric
import torch.nn.functional as F
import numpy as np

from adict import adict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from GraphTranslatorModule import GraphTranslatorModule
from encoders import TimeEncodingOptions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_one_hot(arr):
    n = len(arr)
    x = np.zeros((n,n),dtype=int)
    for i in range(n):
        x[i][arr[i]] = 1
    return torch.Tensor(x )

def edges_conversion(edge_index, n):
    edges = np.zeros((n,n),dtype=int)
    for i in range(edge_index.shape[1]): #remove bidirectional edges to change this
        edges[edge_index[0][i]][edge_index[1][i]] = 1
    return torch.Tensor(edges )

def data_loader_conversion(x,edge_index,y_batch, y_index):
    data_list=[]
    edges = x #todo derive edges from edge index as it can be different    
    n = x.shape[0]
    edges = edges_conversion(edge_index, n)
    nodes = np.identity(edges.shape[-1])
    nodes = torch.tensor(nodes).float()
    y = y_batch[y_index,:] #todo: check this

    y_edges = convert_one_hot( [int(x) for x in y.tolist()] )#categorical to one hot
    y_nodes = np.identity(y_edges.shape[-1])
    y_nodes = torch.tensor(y_nodes).float()
        
    dynamic_edges_mask = np.ones((y_edges.shape))
    dynamic_edges_mask = torch.tensor(dynamic_edges_mask) #todo
    
    time=1 #todo
    time_encoding="sine_informed"
    time_encoder=TimeEncodingOptions()
    context_time = time_encoder(time_encoding)(time)#todo
        
    data_list.append({'edges':edges,
                    'nodes':nodes, 'y_edges':y_edges, 'y_nodes':y_nodes,
                    'dynamic_edges_mask':dynamic_edges_mask,'context_time':context_time})
    
    new_data_loader = DataLoader( data_list,batch_size=1,  )
    return new_data_loader

class GnnConverter(torch.nn.Module):
    def __init__(self, model_configs, y_batch, **kwargs):
        super().__init__()
        checkpoint = kwargs.get('check_point','')
        data_type =  kwargs.get('data_type','')     
        if data_type == 'homer':
            self.graph_model = GraphTranslatorModule.load_from_checkpoint(checkpoint, 
                                                           model_configs = model_configs)
        elif data_type == 'custom': #minimal dataset created
            checkpoint = torch.load(checkpoint)
            self.graph_model = GraphTranslatorModule(model_configs = model_configs)        
            self.graph_model.load_state_dict(checkpoint['model_state_dict'])
        self.y_batch = y_batch    

    def forward(self, x, edge_index, **kwargs):
        y_index = kwargs.get('y_index',0)
        new_train_data_loader = data_loader_conversion(x, edge_index, self.y_batch, y_index)
        data_element = next(iter(new_train_data_loader))         
        _,details,_ = self.graph_model.step(data_element)
        op = torch.log( torch.tensor(details['output_probs']['location'].squeeze()))
        return op

