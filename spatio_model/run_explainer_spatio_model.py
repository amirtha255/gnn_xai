import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def one_hot_to_cat(y):
   y = np.array(y)
   op = np.argmax(y, axis=1)
   return op


def homer_data_to_pyg(data_loader):
    data_list = []
    for step, data_node in enumerate(data_loader):
      x = data_node['edges'].squeeze()
      edge_index = data_node['edges'].squeeze()
      edge_index = np.argwhere(edge_index>0)	     
      y_edges = data_node['y_edges'].squeeze()        
      y = one_hot_to_cat(y_edges)
      data_pyg = Data(x=x,edge_index=edge_index,y=y)
      data_list.append(data_pyg)
    return data_list

if __name__ == '__main__':
    
    with open('spatio_xai_config/xai_spatio_homer.yaml') as f:
      xai_cfg = yaml.safe_load(f)

    if xai_cfg['model_mode']=="training":
      print('training')
      os.system("python3 ./run.py --path={} --name=ours " 
                     " --train_days={} --logs_dir={}/test_checkpoint "
                       "--ckpt_dir={} --read_ckpt".format(xai_cfg['data_dir'], xai_cfg['train_days'],
                                                          xai_cfg['logs_dir'], xai_cfg['ckpt_dir']))
    else:
      with open('config/default.yaml') as f:
          cfg = yaml.safe_load(f)  

      time_options = TimeEncodingOptions(None)
      time_encoding = time_options(cfg['time_encoding'])

      data = RoutinesDataset(data_path=os.path.join(xai_cfg['data_dir'],'processed'), 
                            time_encoder=time_encoding, 
                            batch_size=cfg['batch_size'],
                            max_routines = (xai_cfg['train_days'], None))
      train_data_loader = data.get_train_loader()
      test_data_loader = data.get_test_loader()
      train_data_list = homer_data_to_pyg(train_data_loader)
      test_data_list = homer_data_to_pyg(test_data_loader)      

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

        
      model = GnnConverter(base_model_config, y_batch, check_point=xai_cfg['ckpt'], data_type='homer')
      
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
