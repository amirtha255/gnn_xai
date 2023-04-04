import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer,CaptumExplainer, PGExplainer

import networkx as nx
import numpy as np
import os
import yaml
from adict import adict
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #not show plots interactively

from explainer_wrapper_homer import GnnConverter
from reader import RoutinesDataset
from encoders import TimeEncodingOptions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def one_hot_to_cat(y):
   y = np.array(y)
   op = np.argmax(y, axis=1)
   return torch.tensor(op)


def homer_data_to_pyg(data_loader, flag=False):
    data_list = []
    changed_graphs = []

    for step, data_node in enumerate(data_loader):
      x = data_node['edges'].squeeze().requires_grad_(True)
      edge_index = data_node['edges'].squeeze()
      edge_index = np.argwhere(edge_index>0)	  # not correct as edges are not correctly done
      temp_0 = edge_index[0,:]
      temp_1 = edge_index[1,:]
      edge_index = np.append(edge_index, [[val for val in temp_1], [val for val in temp_0]], 1)     
      edge_index = torch.tensor(edge_index).float().requires_grad_(True)    
      y_edges = data_node['y_edges'].squeeze()        
      y = one_hot_to_cat(y_edges).float().requires_grad_(True)   
      data_pyg = Data(x=x,edge_index=edge_index,y=y, time=data_node['time'],   context_time=data_node['context_time']) #add time and context time here
      data_list.append(data_pyg)  
      if flag:
        if(not(torch.equal(data_node['edges'], data_node['y_edges']))):
         print('Graph changed at ',step, one_hot_to_cat(x.detach().numpy()).float().requires_grad_(True)  , y,data_node['time'] )#,  data_node['context_time'])
         changed_graphs.append(step)
      
    return data_list, changed_graphs

if __name__ == '__main__':
    
    with open('spatio_xai_config/xai_spatio_homer.yaml') as f:
      xai_cfg = yaml.safe_load(f)

    if xai_cfg['model_mode']=="training":
      print('training')
      os.system("python3 ./run.py --path={} --name=ours " 
                     " --train_days={} --logs_dir={}/test_checkpoint ".format(xai_cfg['data_dir'], xai_cfg['train_days'],
                                                          xai_cfg['logs_dir']))
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
      train_data_list, changed_train_graphs = homer_data_to_pyg(train_data_loader, flag=True) #flag to print changed graphs
      test_data_list, changed_test_graphs = homer_data_to_pyg(test_data_loader)      

      base_model_config = {}
      base_model_config['c_len'] = 14 #todo
      base_model_config['edge_importance'] = 'predicted'
      base_model_config['hidden_layer_size'] = 20
      
      graph_index = xai_cfg['graph_index']
      node_index = xai_cfg['node_index']

      data_list = train_data_list
      data_loader = train_data_loader

      if xai_cfg['data_loader_type'] == "test_data_loader":
         data_list = test_data_list
         data_loader = test_data_loader

      base_model_config['n_nodes'] = data_list[0].x.shape[0]
      base_model_config['n_len'] = data_list[0].x.shape[1]        
      base_model_config = adict(base_model_config)
      y_batch = torch.zeros(data_list[0].y.shape)
      for data in data_list:  
        y_batch = torch.vstack((y_batch,torch.tensor(data.y.requires_grad_(True))))
      y_batch = y_batch[1:,:]

      time_batch = torch.zeros(data_list[0].time.shape)
      for data in data_list:
        time_batch = torch.vstack((time_batch,torch.tensor(data.time.requires_grad_(True))))
      time_batch = time_batch[1:,:]

      context_time_batch = torch.zeros(data_list[0].context_time.shape)
      for data in data_list:
        context_time_batch = torch.vstack((context_time_batch,torch.tensor(data.context_time.requires_grad_(True))))
      context_time_batch = context_time_batch[1:,:]
      
      graphs_to_check =  changed_train_graphs # only have the graphs where a node has moved. 
      if xai_cfg['data_loader_type'] == "test_data_loader":
        graphs_to_check = changed_test_graphs

      counterfactual_outputs = {}
      node_ids = { 0:"bathroom",            
          1:"bathroom_counter",            
          2:"home_office",            
              3:"sofa", 
              4: "tvstand",            
              5:"remote_control",         
              6:"toothbrush",            
              7:"toothbrush_holder",              
      }
      data_element = next(iter(data_loader))        

      for graph_index in graphs_to_check:

        counterfactual_outputs[graph_index] = {}
        target = (data_list[graph_index].y) #todo choice in test list
        x, edge_index = data_list[graph_index].x, data_list[graph_index].edge_index #choose which graph to explain      
     
        counterfactual_outputs[graph_index]['x'] = x.argmax(-1)
        counterfactual_outputs[graph_index]['y'] = data_list[graph_index].y.int().detach()
        change_in_input_graph   = np.where(x.argmax(-1) != data_list[graph_index].y )[0]

        model = GnnConverter(base_model_config, y_batch, time_batch, context_time_batch, data_element['dynamic_edges_mask'], check_point=xai_cfg['ckpt'], data_type='homer')
        original_op = model(x,edge_index,y_index = graph_index).argmax(-1)

        nodes_to_monitor = change_in_input_graph # [7]
        counterfactual_outputs[graph_index]['nodes_moved_and_eval'] = nodes_to_monitor        
        counterfactual_outputs[graph_index]['original_op'] = original_op
        counterfactual_outputs[graph_index]['time_instance'] = data_list[graph_index].time
        
        #counterfactuals with time
        counterfactual_outputs[graph_index]['time_changes'] = []
        for i in range( max(0, graph_index-3), min(len(data_list) -1, graph_index+4)): # or we can directly get time
          new_index = i
          model_op = model(x,edge_index,y_index = new_index).argmax(-1)
          changes_in_pred = np.where(model_op!=original_op)[0]

          if len(changes_in_pred):
              for node in nodes_to_monitor:
                 if node in changes_in_pred:
                  #print(' model op with changed time {} is'.format(new_index - graph_index), model_op) #CHECK
                  #print(node_ids[int(model_op[node].numpy())])                  
                  counterfactual_outputs[graph_index]['time_changes'].append(( new_index - graph_index ))
              

        # counterfactuals with changing node values only for dyn edges
        x = x.clone()
        
        counterfactual_outputs[graph_index]['edge_changes'] = []
        for node in nodes_to_monitor: # dervie from dyn edges mask or searchable..
          # set to zero other 1's for those nodes
          initial_location = x.argmax(-1)[node].detach().numpy()
          x[node][initial_location] = 0

          for i in range(len(x[0])):         
            x[node][i] = 1 #should I uncheck one in prev value
            model_op = model(x,edge_index,y_index = graph_index).argmax(-1)   
            changes_in_pred = np.where(model_op!=original_op)[0]
            if len(changes_in_pred):
              if node in changes_in_pred:
                counterfactual_outputs[graph_index]['edge_changes'].append((node,i))
                #print(' new x changed model op for x[{}][{}] is'.format(node,i), model_op)
            x[node][i] = 0
          x[node][initial_location] = 1


      for key,val in counterfactual_outputs.items():
        print('Graph index checked', key)
        for k,v in val.items():
          print(k,v)
        print('\n')
