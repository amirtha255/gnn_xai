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

from homer_explainer_wrapper import GnnConverter
from reader import RoutinesDataset
from encoders import TimeEncodingOptions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def one_hot_to_cat(y):
   y = np.array(y)
   op = np.argmax(y, axis=1)
   return op


def homer_data_to_pyg(data_loader):
    data_list = []
    #data_train_node= next(iter(data_loader))	
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

    data_dir="/workspace/SpatioTemporalObjectTracking/data/HOMER/household0/"
    logs_dir="training_logs"
    train_days=2
    ckpt_dir="/workspace/SpatioTemporalObjectTracking/training_logs/1/ours_50epochs/"
    ckpt="/workspace/SpatioTemporalObjectTracking/training_logs/1/ours_50epochs/epoch=49-step=5400.ckpt"
    mode="explain"
    data_type = "training" # "test"
    if mode=="training":
       print('training')
      #training model
      #os.system("python3 ./homer_run.py --path={} --name=ours " 
      #               " --train_days={} --logs_dir={}/test_checkpoint "
      #                 "--ckpt_dir={} --read_ckpt".format(dataset, train_days, logs_dir, ckpt_dir))
    else:
      with open('config/default.yaml') as f:
          cfg = yaml.safe_load(f)  

      time_options = TimeEncodingOptions(None)
      time_encoding = time_options(cfg['time_encoding'])

      data = RoutinesDataset(data_path=os.path.join(data_dir,'processed'), 
                            time_encoder=time_encoding, 
                            batch_size=cfg['batch_size'],
                            max_routines = (train_days, None))
      train_data_loader = data.get_train_loader()
      test_data_loader = data.get_test_loader()
      train_data_list = homer_data_to_pyg(train_data_loader)
      test_data_list = homer_data_to_pyg(test_data_loader)
      

      dict_node_num_to_class = {}
      for i in range(len(data.node_classes)):
          dict_node_num_to_class[i] = data.node_classes[i]

      model_configs = {}
      # for custom data
      if data_type=="training":
        model_configs['n_nodes'] = train_data_list[0].x.shape[0]
        model_configs['n_len'] = train_data_list[0].x.shape[1]
        model_configs['c_len'] = 14 #todo
        model_configs['edge_importance'] = 'predicted'
        model_configs['hidden_layer_size'] = 20
        model_configs = adict(model_configs)


        y_batch = torch.zeros(train_data_list[0].y.shape)
        for data in train_data_list: #todo we only ask for explanations from this list 
          y_batch = torch.vstack((y_batch,torch.tensor(data.y)))
        y_batch = y_batch[1:,:]
      
      else:
        model_configs['n_nodes'] = test_data_list[0].x.shape[0]
        model_configs['n_len'] = test_data_list[0].x.shape[1]
        model_configs['c_len'] = 14 #todo
        model_configs['edge_importance'] = 'predicted'
        model_configs['hidden_layer_size'] = 20
        model_configs = adict(model_configs)

        y_batch = torch.zeros(11)
        for data in test_data_list: #todo we only ask for explanations from this list 
            y_batch = torch.vstack((y_batch,data.y))   
        y_batch = y_batch[1:,:]
        
      model = GnnConverter(model_configs, y_batch, check_point=ckpt)
      
      graph_index =10
      node_index = 2
      x, edge_index = train_data_list[graph_index].x.to(device),train_data_list[graph_index].edge_index.to(device) #choose which graph to explain
      op = torch.Tensor(train_data_list[graph_index].y).to(device)
      op = op.long()

      explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='phenomenon',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
            ),
      )   

      explanation = explainer(x, edge_index, index=node_index, y_index=node_index, target=op)
      print(f'Generated explanations in {explanation.available_explanations}')

      path = 'feature_importance.png'
      explanation.visualize_feature_importance(path, top_k=5)
      print(f"Feature importance plot has been saved to '{path}'")

      path = 'subgraph.pdf'
      explanation.visualize_graph(path)
      print(f"Subgraph visualization plot has been saved to '{path}'")