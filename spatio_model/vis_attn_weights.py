import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer,CaptumExplainer, PGExplainer
from torch_geometric.visualization import visualize_graph
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from adict import adict
import networkx as nx
import numpy as np
import pandas as pd
import os
import sys
import yaml
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #not show plots interactively

sys.path.append('helpers')
from encoders import TimeEncodingOptions
from GraphTranslatorModule import GraphTranslatorModule

import wandb
from pytorch_lightning.loggers import WandbLogger

#todo across more hops
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
    edges = x.clone().detach() 
    nodes = np.identity(edges.shape[-1])
    nodes = torch.tensor(nodes).float()
    y = y_batch[y_index,:] #todo: check this
    
    y_edges = convert_one_hot( [int(x) for x in y.tolist()] )#categorical to one hot
    y_nodes = np.identity(y_edges.shape[-1])
    y_nodes = torch.tensor(y_nodes).float()
        
    dynamic_edges_mask = np.ones((y_edges.shape))
    dynamic_edges_mask = torch.tensor(dynamic_edges_mask) #todo
    
    time=(y_index+1)%13 #todo map from dataset to time
    time_encoding="sine_informed"
    time_encoder=TimeEncodingOptions()
    context_time = time_encoder(time_encoding)(time)#todo
            
    data_list.append({'edges':edges.requires_grad_(True),
                    'nodes':nodes.requires_grad_(True), 'y_edges':y_edges.requires_grad_(True), 'y_nodes':y_nodes.requires_grad_(True),
                    'dynamic_edges_mask':dynamic_edges_mask.requires_grad_(True),'context_time':context_time.requires_grad_(True)})
    new_data_loader = DataLoader( data_list,batch_size=1,  )
    return new_data_loader

class ImpFeatures(torch.nn.Module):
    def __init__(self, model_configs, y_batch, **kwargs):
        super().__init__()
        checkpoint = kwargs.get('check_point','')
        data_type =  kwargs.get('data_type','')       
        self.y_batch = y_batch  
        if data_type == 'homer':
            self.graph_model = GraphTranslatorModule.load_from_checkpoint(checkpoint, 
                                                           model_configs = model_configs)
        elif data_type == 'custom': #minimal dataset created
            checkpoint = torch.load(checkpoint)
            self.graph_model = GraphTranslatorModule(model_configs = model_configs)        
            self.graph_model.load_state_dict(checkpoint['model_state_dict'])
          

    def forward(self, x, edge_index, **kwargs):

        y_index = kwargs.get('y_index') #should be called graph index 
        if not y_index:
            y_index = 0
        new_train_data_loader = data_loader_conversion(x, edge_index, self.y_batch, y_index)
        data_element = next(iter(new_train_data_loader))         
        edges,_,_,imp = self.graph_model(data_element['edges'], data_element['nodes'], data_element['context_time'])
        """
        print('\n x is ',x)
        print('\n edge index is ',edge_index)
        print('\n y index is ',y_index)
        print('\n converted edges is ',data_element['edges'])
        print('\n converted nodes is ',data_element['nodes'])
        """
        return imp.squeeze()
        


def spatio_explanations(xai_cfg):
      train_data_list = torch.load(xai_cfg['train_data_loader'])
      test_data_list = torch.load(xai_cfg['test_data_loader'])   

      base_model_config = {}
      base_model_config['c_len'] = 14 #todo
      base_model_config['edge_importance'] = 'predicted'
      base_model_config['hidden_layer_size'] = 20
      base_model_config['n_nodes'] = train_data_list[0].x.shape[0]
      base_model_config['n_len'] = train_data_list[0].x.shape[1]        
      base_model_config = adict(base_model_config)
      
      graph_index = xai_cfg['graph_index']
      # to explain edge e_ij, node_index = i and end_node_index = j
      node_index = xai_cfg['node_index']

      if xai_cfg['data_loader_type'] == "train_data_loader":
        y_batch = torch.zeros(train_data_list[0].y.shape)
        for data in train_data_list: #todo we only ask for explanations from this list 
          y_batch = torch.vstack((y_batch,torch.tensor(data.y)))
        y_batch = y_batch[1:,:]
        target = (train_data_list[graph_index].y.long()) #todo choice in test list
        x, edge_index = train_data_list[graph_index].x.to(device),train_data_list[graph_index].edge_index.to(device) #choose which graph to explain
      
      else:
        y_batch = torch.zeros(test_data_list[0].y.shape)
        for data in test_data_list: #todo we only ask for explanations from this list 
            y_batch = torch.vstack((y_batch,data.y))   
        y_batch = y_batch[1:,:]

        target = (test_data_list[graph_index].y.long()) #todo choice in test list
        x, edge_index = test_data_list[graph_index].x.to(device),test_data_list[graph_index].edge_index.to(device) #choose which graph to explain

        
      model = ImpFeatures(base_model_config, y_batch, check_point=xai_cfg['ckpt'], data_type='custom')
      imp = model(x, edge_index, y_index=graph_index)
      return x, imp
      

def plot_all_imp_one_hop(adj, imp, hops, top_k,plot_wrt_direction  ):
    # k hop - Adj power k multipled with imp
    
    node_i = xai_cfg['node_index'] #i
    node_j = xai_cfg['end_node_index'] #j
    #correct_end_node_index = train_data_list[graph_index].y[node_index]
    
    # 1 hop
    imp_starts_at_i = list(imp[node_i,:].detach().numpy())
    imp_starts_at_j = list(imp[node_j,:].detach().numpy())
    imp_ends_at_i = list(imp[:,node_i].detach().numpy())
    imp_ends_at_j = list(imp[:, node_j].detach().numpy())

    # set width of bar
    barWidth = 0.1
    fig = plt.subplots(figsize =(12, 8))
    
    # Set position of bar on X axis
    br1 = np.arange(len(imp_starts_at_i))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    
    # Make the plot
    plt.bar(br1, imp_starts_at_i, color ='r', width = barWidth,
            edgecolor ='grey', label ='imp_starts_at_i {}'.format(node_i))
    plt.bar(br2, imp_starts_at_j, color ='g', width = barWidth,
            edgecolor ='grey', label ='imp_starts_at_j {}'.format(node_j))
    plt.bar(br3, imp_ends_at_i, color ='b', width = barWidth,
            edgecolor ='grey', label ='imp_ends_at_i {}'.format(node_i))
    plt.bar(br4, imp_ends_at_j, color ='y', width = barWidth,
            edgecolor ='grey', label ='imp_ends_at_j {}'.format(node_j))
    
    # Adding Xticks   
    plt.xticks([r + barWidth for r in range(len(imp_starts_at_i))],
            [str(r) for r in range(len(imp_starts_at_i)) ])
    
    plt.legend()
    plt.xlabel("Node values ")
    plt.ylabel("Attention values")
    plt.title("Plot of importances wrt node {}".format(node_i))
    plt.show()
    plt.savefig(os.path.join(xai_cfg['output_dir'],'plot_all_imp_one_hop.png'), bbox_inches='tight')

    

def plot_most_imp(adj, imp, hops, top_k,plot_wrt_direction  ):
    # k hop - Adj power k multipled with imp

    node_i = xai_cfg['node_index'] #i
    node_j = xai_cfg['end_node_index'] #j
    #correct_end_node_index = train_data_list[graph_index].y[node_index]
    # 1 hop
    imp_starts_at_i = list(imp[node_i,:].detach().numpy())
    imp_starts_at_j = list(imp[node_j,:].detach().numpy())
    imp_ends_at_i = list(imp[:,node_i].detach().numpy())
    imp_ends_at_j = list(imp[:, node_j].detach().numpy())

    fig, axes = plt.subplots(nrows=2, ncols=2)
    feat_labels = [str(r) for r in range(len(imp_starts_at_i)) ]
    
    df1 = pd.DataFrame({'feat_importance': imp_starts_at_i},
                          index=feat_labels)
    df1 = df1.sort_values("feat_importance", ascending=False)
    #df = df.round(decimals=3)
    if top_k is not None:
        df1 = df1.head(top_k)
        title = f"Feature importance for top {len(df1)} features, where node starts at i -  {node_i}"        
    else:
        title = f"Feature importance for {len(df1)} features, where node starts at i - {node_i}"
    ax1 = df1.plot(
            ax=axes[0,0],
            kind='barh',
            figsize=(15, 10),
            title=title,
            ylabel='End node',
            xlim=[0, float(max(imp_starts_at_i)) + 1],
            legend=False,
        )
    plt.gca().invert_yaxis()
    ax1.bar_label(container=ax1.containers[0], label_type='edge')  

    df2 = pd.DataFrame({'feat_importance': imp_starts_at_j},
                          index=feat_labels)
    df2 = df2.sort_values("feat_importance", ascending=False)
    #df = df.round(decimals=3)
    if top_k is not None:
        df2 = df2.head(top_k)
        title = f"Node starts at j - {node_j}"        
    else:
        title = f"Node starts at j - {node_j}"
    ax2 = df2.plot(
            ax=axes[0,1],
            kind='barh',
            figsize=(15,10),
            title=title,
            ylabel='End node',
            xlim=[0, float(max(imp_starts_at_j)) + 1],
            legend=False,
        )    
    plt.gca().invert_yaxis()
    ax2.bar_label(container=ax2.containers[0], label_type='edge')    

    df3 = pd.DataFrame({'feat_importance': imp_ends_at_i},
                          index=feat_labels)
    df3 = df3.sort_values("feat_importance", ascending=False)
    #df = df.round(decimals=3)
    if top_k is not None:
        df3 = df3.head(top_k)
        title = f"Feature importance for top {len(df3)} features, where node ends at i -  {node_i}"      
    else:
        title = f"Feature importance for {len(df3)} features, where node ends at i - {node_i}"
    ax3 = df3.plot(
            ax=axes[1,0],
            kind='barh',
            figsize=(15, 10),
            title=title,
            ylabel='Start node',
            xlim=[0, float(max(imp_ends_at_i)) + 1],
            legend=False,
        )
    plt.gca().invert_yaxis()
    ax3.bar_label(container=ax3.containers[0], label_type='edge') 

    df4 = pd.DataFrame({'feat_importance': imp_ends_at_j},
                          index=feat_labels)
    df4 = df4.sort_values("feat_importance", ascending=False)
    
    #df = df.round(decimals=3)
    if top_k is not None:
        df4 = df4.head(top_k)
        title = f"node ends at j -  {node_j}"      
    else:
        title = f"node ends at j - {node_j}"
    ax4 = df4.plot(
            ax=axes[1,1],
            kind='barh',
            figsize=(15, 10),
            title=title,
            ylabel='Start node',
            xlim=[0, float(max(imp_ends_at_j)) + 1],
            legend=False,
        )
    plt.gca().invert_yaxis()
    ax4.bar_label(container=ax4.containers[0], label_type='edge') 
    
    plt.savefig(os.path.join(xai_cfg['output_dir'],'plot_most_imp.png'), bbox_inches='tight')


def plot_most_imp_across_all(adj, imp, hops, top_k,plot_wrt_direction  ):
    node_i = xai_cfg['node_index'] #i
    node_j = xai_cfg['end_node_index'] #j

    imp_starts_at_i = list(imp[node_i,:].detach().numpy())
    imp_starts_at_j = list(imp[node_j,:].detach().numpy())
    imp_ends_at_i = list(imp[:,node_i].detach().numpy())
    imp_ends_at_j = list(imp[:, node_j].detach().numpy())

    label_starts_at_i = [ str(node_i) + ',' + str(val) for val in range(len(imp_starts_at_i))]
    label_starts_at_j = [ str(node_j) + ',' + str(val) for val in range(len(imp_starts_at_i))]
    label_ends_at_i = [ str(val) for val in range(len(imp_starts_at_i)) ] #+ ',' + str(node_i)   ]    
    label_ends_at_j = [str(val) for val in range(len(imp_starts_at_i)) ]
    for ind in range(len(label_ends_at_i)):
        label_ends_at_i[ind] += ',' + str(node_i)  
    for ind in range(len(label_ends_at_j)):
        label_ends_at_j[ind] += ',' + str(node_j)   
    
    imp_all_directions = imp_starts_at_i
    imp_all_directions = np.append(imp_all_directions, imp_starts_at_j)
    imp_all_directions = np.append(imp_all_directions, imp_ends_at_i)
    imp_all_directions = np.append(imp_all_directions, imp_ends_at_j)
    labels = label_starts_at_i
    labels = np.append(labels, label_starts_at_j)
    labels = np.append(labels, label_ends_at_i)
    labels = np.append(labels, label_ends_at_j)
    
    df = pd.DataFrame({'feat_importance': imp_all_directions},
                          index=labels)
    df = df.sort_values("feat_importance", ascending=False)
    df = df.round(decimals=3)

    if top_k is not None:
        df = df.head(top_k)
        title = f"Feature importance for top {len(df)} features"
    else:
        title = f"Feature importance for {len(df)} features"

    ax = df.plot(
            kind='barh',
            figsize=(10, 7),
            title=title,
            ylabel='Feature label',
            xlim=[0, float(imp_all_directions.max()) + 0.3],
            legend=False,
    )
    plt.gca().invert_yaxis()
    ax.bar_label(container=ax.containers[0], label_type='edge')
    plt.savefig(os.path.join(xai_cfg['output_dir'],'plot_most_imp_across_all.png'), bbox_inches='tight')



def plot_one_direction(adj, imp, hops, top_k,plot_wrt_direction  ):
    node_i = xai_cfg['node_index'] #i
    node_j = xai_cfg['end_node_index'] #j
    attn_values = list(imp[node_i,:].detach().numpy())
    num_nodes = list([i for i in range(len(imp[node_i]))])
    
    fig = plt.figure(figsize = (10, 5))    
    # creating the bar plot
    plt.bar(num_nodes, attn_values, color ='maroon',
            width = 0.4)    
        
    plt.xlabel("Edges ")
    plt.ylabel("Attention values")
    plt.title("Attention importance for {} hops, top {}, plot_wrt_direction {} for node {} with prediction {} and correct target {}".
                format(hops, top_k,plot_wrt_direction, node_i,node_j , node_j )) # change later
    plt.show()
    plt.savefig(os.path.join(xai_cfg['output_dir'],'plot_one_direction.png'), bbox_inches='tight')


if __name__ == '__main__':    
    with open('spatio_xai_config/xai_spatio_custom.yaml') as f:
      xai_cfg = yaml.safe_load(f)
    adj, imp = spatio_explanations(xai_cfg)
    plot_all_imp_one_hop(adj, imp, xai_cfg['hops'], xai_cfg['top_k'], xai_cfg['plot_wrt_direction'])
    plot_most_imp(adj, imp, xai_cfg['hops'], xai_cfg['top_k'], xai_cfg['plot_wrt_direction'])
    plot_one_direction(adj, imp, xai_cfg['hops'], xai_cfg['top_k'], xai_cfg['plot_wrt_direction'])
    plot_most_imp_across_all(adj, imp, xai_cfg['hops'], xai_cfg['top_k'], xai_cfg['plot_wrt_direction'])


