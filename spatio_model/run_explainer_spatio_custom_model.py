import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import networkx as nx
import numpy as np
import os
import yaml
from adict import adict
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #not show plots interactively

from explainer_wrapper import GnnConverter
from reader import RoutinesDataset
from encoders import TimeEncodingOptions
from GraphTranslatorModule import GraphTranslatorModule


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def one_hot_to_cat(y):
   y = np.array(y)
   op = np.argmax(y, axis=1)
   return op


def convert_one_hot(arr):
    # return one hot encoding as x given the target representation, y
    n = len(arr)
    x = np.zeros((n,n),dtype=int)
    for i in range(n):
        x[i][arr[i]] = 1
    return torch.Tensor(x )#, dtype=torch.float)

def data_loader_conversion(input_data_loader):
    data_list=[]
    time=0 # need to split at each 20 time steps
    for data_node in input_data_loader:       	

        edges = data_node.x
        #edges = edges.unsqueeze(0)
        
        nodes = np.identity(edges.shape[-1])
        #nodes = torch.tensor(np.expand_dims(nodes,axis=0)) #todo
        nodes = torch.tensor(nodes).float()

        y_edges = convert_one_hot( [int(x) for x in data_node.y.tolist()] )#categorical to one hot
        #y_edges = y_edges.unsqueeze(0)

        y_nodes = np.identity(y_edges.shape[-1])
        y_nodes = torch.tensor(y_nodes).float()
        #y_nodes = torch.tensor(np.expand_dims(nodes,axis=0)) #todo

        dynamic_edges_mask = np.ones((y_edges.shape))
        dynamic_edges_mask = torch.tensor(dynamic_edges_mask) #todo

        time+=1
        time = time%20
        time_encoding="sine_informed"
        time_encoder=TimeEncodingOptions()
        context_time = time_encoder(time_encoding)(time)#todo
        

        data_list.append({'edges':edges,
                    'nodes':nodes, 'y_edges':y_edges, 'y_nodes':y_nodes,
                    'dynamic_edges_mask':dynamic_edges_mask,'context_time':context_time})

    print('size of data list is ',len(data_list))
    new_data_loader = DataLoader( data_list,batch_size=1, num_workers=8, )
    return new_data_loader

def create_custom_data(data_path):

    model_configs={}    
    train_data_list   = torch.load(data_path)    
    #needed to graphtranslator module
    model_configs['n_nodes'] = train_data_list[0].x.shape[0]
    model_configs['n_len'] = train_data_list[0].x.shape[1]
    model_configs['c_len'] = 14 #todo
    model_configs['edge_importance'] = 'predicted'
    model_configs['hidden_layer_size'] = 20
    model_configs = adict(model_configs)
    return model_configs
    

if __name__ == '__main__':
    
    with open('xai_spatio_custom.yaml') as f:
      xai_cfg = yaml.safe_load(f)

    if xai_cfg['model_mode']=="training":
      base_model_configs =  create_custom_data(xai_cfg['train_data_loader'])
    
      train_data_loader = DataLoader(torch.load(xai_cfg['train_data_loader']))
      test_data_loader = DataLoader(torch.load(xai_cfg['test_data_loader']))
      new_train_data_loader = data_loader_conversion(train_data_loader)
      new_test_data_loader = data_loader_conversion(test_data_loader)
      model = GraphTranslatorModule(model_configs = base_model_configs)
      epochs = [1]
      done_epochs = 0
      output_dir_new="training_logs/check_new_model"
      for epoch in epochs:          
          trainer = Trainer( max_epochs=epoch-done_epochs, log_every_n_steps=5)#gpus = torch.cuda.device_count(),
          trainer.fit(model, new_train_data_loader)
          trainer.test(model, new_test_data_loader)
          
      PATH = "spatio_temporal_on_custom.pt"    
      torch.save({
            'model_state_dict': model.state_dict(),
            }, PATH)
      
    else:
      
      train_data_list = torch.load(xai_cfg['train_data_loader'])
      test_data_list = torch.load(xai_cfg['test_data_loader'])   

      base_model_config = {}
      base_model_config['c_len'] = 14 #todo
      base_model_config['edge_importance'] = 'predicted'
      base_model_config['hidden_layer_size'] = 20
      
      graph_index = xai_cfg['graph_index']
      node_index = xai_cfg['node_index']
      # for custom data
      if xai_cfg['data_loader_type'] == "train_data_loader":
        base_model_config['n_nodes'] = train_data_list[0].x.shape[0]
        base_model_config['n_len'] = train_data_list[0].x.shape[1]        
        base_model_config = adict(base_model_config)
        y_batch = torch.zeros(train_data_list[0].y.shape)
        for data in train_data_list: #todo we only ask for explanations from this list 
          y_batch = torch.vstack((y_batch,torch.tensor(data.y)))
        y_batch = y_batch[1:,:]

        target = (train_data_list[graph_index].y) #todo choice in test list
        x, edge_index = train_data_list[graph_index].x.to(device),train_data_list[graph_index].edge_index.to(device) #choose which graph to explain

      
      else:
        base_model_config['n_nodes'] = test_data_list[0].x.shape[0]
        base_model_config['n_len'] = test_data_list[0].x.shape[1]
        base_model_config = adict(base_model_config)
        y_batch = torch.zeros(test_data_list[0].y.shape)
        for data in test_data_list: #todo we only ask for explanations from this list 
            y_batch = torch.vstack((y_batch,data.y))   
        y_batch = y_batch[1:,:]

        target = (test_data_list[graph_index].y) #todo choice in test list
        x, edge_index = test_data_list[graph_index].x.to(device),test_data_list[graph_index].edge_index.to(device) #choose which graph to explain

        
      model = GnnConverter(base_model_config, y_batch, check_point=xai_cfg['ckpt'], data_type='custom')
      
      model_config=dict(
            mode=xai_cfg['mode'],
            task_level=xai_cfg['task_level'],
            return_type=xai_cfg['return_type'],
          )
    
      algorithm=None
      if xai_cfg['algorithm'] == 'GNNExplainer':
          algorithm = GNNExplainer(epochs=xai_cfg['explanation_epochs'])
      elif xai_cfg['algorithm'] == 'PGExplainer':
          algorithm = PGExplainer(epochs=xai_cfg['explanation_epochs'])
      elif xai_cfg['algorithm'] == 'CaptumExplainer':
          algorithm = CaptumExplainer(attribution_method='Integrated_Gradients')
      else:
          print('not a valid explanation algorithm')
          exit(0)

      explainer = Explainer(
          model=model,
          algorithm=algorithm,
          explanation_type=xai_cfg['explanation_type'],
          node_mask_type=xai_cfg['node_mask_type'],
          edge_mask_type=xai_cfg['edge_mask_type'],
          model_config=model_config,
      )
         
      explanation = explainer(x, edge_index, index=node_index, target=target)
      print(f'Generated explanations in {explanation.available_explanations}')

      if not os.path.exists(xai_cfg['output_dir']):
          os.makedirs(xai_cfg['output_dir'])

      path = os.path.join(xai_cfg['output_dir'],'feature_importance_g{}_n{}_{}.png'.format(graph_index,node_index,xai_cfg['name']))
      explanation.visualize_feature_importance(path, top_k=5)
      print(f"Feature importance plot has been saved to '{path}'")

      path = os.path.join(xai_cfg['output_dir'],'subgraph_g{}_n{}_{}.pdf'.format(graph_index,node_index,xai_cfg['name']))
      explanation.visualize_graph(path)
      print(f"Subgraph visualization plot has been saved to '{path}'")
