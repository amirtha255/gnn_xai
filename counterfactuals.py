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
import os
import sys
import yaml
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #not show plots interactively

from create_dataset import y_to_edge, y_to_x
from explainer_wrapper import convert_one_hot
from explainer_wrapper import GnnConverter
from reader import RoutinesDataset
from encoders import TimeEncodingOptions
from GraphTranslatorModule import GraphTranslatorModule

import wandb
from pytorch_lightning.loggers import WandbLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node_ids_from_classes = {0:'kitchen',1:'fridge',2:'counter',3:'cabinet',4:'milk',5:'cereal', 6:'coffee',
                             7:'keys', 8:'cup', 9:'bowl', 10:'rack'}

def display_node_ids(list_to_node_classes):
  
  if list_to_node_classes.size:
    print([node_ids_from_classes[ind] + '-> '+node_ids_from_classes[val.item()] for ind,val in enumerate(list_to_node_classes)])


def change_time(graph_at_origin_time, train_data_list, model):
      graph_at_origin_time = 237
      node_to_check = 4
      x, edge_index = train_data_list[graph_at_origin_time].x.to(device),train_data_list[graph_at_origin_time].edge_index.to(device) #here
      original_pred = model(x, edge_index, y_index=graph_at_origin_time).argmax(-1)
      #print(original_pred, pred, np.where(original_pred!=pred))
      print('Original prediction at time {} is {}'.format(graph_at_origin_time, original_pred))
      
      t = graph_at_origin_time % 13
      for i in range(t,0,-1):
        time_stamp = graph_at_origin_time - i
        pred  = model(x, edge_index, y_index=time_stamp).argmax(-1)

        #print('Difference in predictions between origin and time stamp t {} is {}'.
              #format(time_stamp-graph_at_origin_time,  np.where(original_pred!=pred)))
        for val in np.where(original_pred!=pred)[0]:
           val = val.item()
           if(val == node_to_check):
            print('At t '+ str(time_stamp-graph_at_origin_time),' original pred is ', node_ids_from_classes[val] + '->' + node_ids_from_classes[original_pred[val].item()], ', \n new pred is ',node_ids_from_classes[val] + '->' + node_ids_from_classes[pred[val].item()])
      
      for i in range(t+1,14):
        time_stamp = graph_at_origin_time + i - t
        pred  = model(x, edge_index, y_index=time_stamp).argmax(-1)
        #print('Difference in predictions between origin and time stamp t + {} is {}'.
              #format(time_stamp-graph_at_origin_time,  np.where(original_pred!=pred)))
        
        for val in np.where(original_pred!=pred)[0]:
           val = val.item()
           if(val == node_to_check):
            print('At t + '+ str(time_stamp-graph_at_origin_time),' original pred is ', node_ids_from_classes[val] + '->' + node_ids_from_classes[original_pred[val].item()], ', \n new pred is ',node_ids_from_classes[val] + '->' + node_ids_from_classes[pred[val].item()])
      
           #print('original is ', node_ids_from_classes[val] + '->' + node_ids_from_classes[original_pred[val].item()], ', new is ',node_ids_from_classes[val] + '->' + node_ids_from_classes[pred[val].item()])
        #print(display_node_ids(np.where(original_pred!=pred)[0]))

def change_graph_at_time(graph_at_origin_time, train_data_list, model):
      graph_at_origin_time = 1
      node_to_check = 6
      x, edge_index = train_data_list[graph_at_origin_time].x.to(device),train_data_list[graph_at_origin_time].edge_index.to(device) #here
      original_pred = model(x, edge_index, y_index=graph_at_origin_time).argmax(-1)
      print('Original graph at time {} is {}'.format(graph_at_origin_time, x.argmax(-1)))
      
      t = graph_at_origin_time % 13
      for i in range(t,0,-1):
        time_stamp = graph_at_origin_time - i
        #print(time_stamp)
        x, edge_index = train_data_list[time_stamp].x.to(device),train_data_list[time_stamp].edge_index.to(device) #here        
        pred  = model(x, edge_index, y_index=graph_at_origin_time).argmax(-1)
        #print('Difference in predictions between origin and time stamp t {} is {}'.
              #format(time_stamp-graph_at_origin_time,  np.where(original_pred!=pred)))        
        for val in np.where(original_pred!=pred)[0]:
           val = val.item()
           if(val == node_to_check):
            print('x is ',x.argmax(-1))
            #print(x.argmax(-1))
            print('Input graph at step at t '+ str(time_stamp-graph_at_origin_time),' original pred is ', node_ids_from_classes[val] + '->' + node_ids_from_classes[original_pred[val].item()], ', new pred is ',node_ids_from_classes[val] + '->' + node_ids_from_classes[pred[val].item()])
      scene = torch.Tensor([0,  0,  0,  0,  2,  3,  2,  2,  2, 10,  0])  # coffee is on counter, what happends to cup
      temp_y = [int(x) for x in scene.tolist()]
      x_scene = y_to_x(temp_y)
      edge_scene = y_to_edge(temp_y)
      pred  = model(x_scene, edge_scene, y_index=graph_at_origin_time).argmax(-1)
      for val in np.where(original_pred!=pred)[0]:
           val = val.item()
           if(val == node_to_check):
            print('***** x is ',x.argmax(-1))


def change_graph(graph_at_origin_time, train_data_list, model):
      x, edge_index = train_data_list[graph_at_origin_time].x.to(device),train_data_list[graph_at_origin_time].edge_index.to(device) #here
      original_pred = model(x, edge_index, y_index=graph_at_origin_time).argmax(-1)
      #print(original_pred, pred, np.where(original_pred!=pred))
      print('Original prediction at time {} is {}'.format(graph_at_origin_time, original_pred))
      """
      n = len(x)
      temp_val = x.argmax(-1)
      print('original graph x is ',temp_val, edge_index)
      for i in range(n): #how to toggle
         print(temp_val[i]^0)
      #use temp val to toggle, get new x and edge index
      """
      t = graph_at_origin_time % 13
      node_to_check = 8
      for i in range(t,0,-1):
        time_stamp = graph_at_origin_time - i
        print('grah at ',time_stamp)
        x, edge_index = train_data_list[time_stamp].x.to(device),train_data_list[time_stamp].edge_index.to(device) #here      
        pred  = model(x, edge_index, y_index=graph_at_origin_time).argmax(-1)

        print('Difference in predictions  {} is {}'.
              format(time_stamp-graph_at_origin_time,  np.where(original_pred!=pred)))
        for val in np.where(original_pred!=pred)[0]:
           val = val.item()

           print('original is ', node_ids_from_classes[val] + '->' + node_ids_from_classes[original_pred[val].item()], ', new is ',node_ids_from_classes[val] + '->' + node_ids_from_classes[pred[val].item()])
      
      
def bfs_search(graph_at_origin_time, train_data_list, model):
      graph_at_origin_time = 1
      node_to_check = 6
      x, edge_index = train_data_list[graph_at_origin_time].x.to(device),train_data_list[graph_at_origin_time].edge_index.to(device) #here
      original_pred = model(x, edge_index, y_index=graph_at_origin_time).argmax(-1)
      
      
      t = graph_at_origin_time % 13
      # change graph by switching the target of each node - but how to impose prior
      scene = torch.Tensor(x.argmax(-1))  # coffee is on counter, what happends to cup
      temp_scene = scene.clone()

      #how many hops
      for i in range(len(temp_scene)):
            temp_scene[i] = original_pred[node_to_check]
            temp_y = [int(x) for x in temp_scene.tolist()]
            x_scene = y_to_x(temp_y)
            edge_scene = y_to_edge(temp_y)
            pred  = model(x_scene, edge_scene, y_index=graph_at_origin_time).argmax(-1)
            print(i, pred,original_pred)



      #for val in np.where(original_pred!=pred)[0]:           
      #      print('Input graph at step at t '+ str(time_stamp-graph_at_origin_time),' original pred is ', node_ids_from_classes[val] + '->' + node_ids_from_classes[original_pred[val].item()], ', new pred is ',node_ids_from_classes[val] + '->' + node_ids_from_classes[pred[val].item()])



if __name__ == '__main__':
    
      with open('spatio_xai_config/xai_spatio_custom.yaml') as f:
        xai_cfg = yaml.safe_load(f)
      train_data_list = torch.load(xai_cfg['train_data_loader'])
      test_data_list = torch.load(xai_cfg['test_data_loader'])   

      base_model_config = {}
      base_model_config['c_len'] = 14 #todo
      base_model_config['edge_importance'] = 'predicted'
      base_model_config['hidden_layer_size'] = 20
      base_model_config['n_nodes'] = train_data_list[0].x.shape[0]
      base_model_config['n_len'] = train_data_list[0].x.shape[1]        
      base_model_config = adict(base_model_config)
      
      graph_index = 248 #here 
      origin_graph_index = 246
      
      y_batch = torch.zeros(train_data_list[0].y.shape)
      for data in train_data_list: #todo we only ask for explanations from this list 
        y_batch = torch.vstack((y_batch,torch.tensor(data.y)))
      y_batch = y_batch[1:,:]
      target = (train_data_list[graph_index].y.long()) #todo choice in test list
      x, edge_index = train_data_list[graph_index].x.to(device),train_data_list[graph_index].edge_index.to(device) #choose which graph to explain
    
      x, edge_index = train_data_list[origin_graph_index].x.to(device),train_data_list[origin_graph_index].edge_index.to(device) #here
     
      model = GnnConverter(base_model_config, y_batch, check_point=xai_cfg['ckpt'], data_type='custom')
      pred  = model(x, edge_index, y_index=graph_index)      
      
      graph_index = 2 #here 
      origin_graph_index = 4

      #change_graph_at_time(2, train_data_list, model)
      #change_time(2, train_data_list, model)

      #change_graph(9, train_data_list, model)
      
      #print(train_data_list[origin_graph_index].y, y_batch[origin_graph_index])
      #print('graph index is ', graph_index,' and origin graph index is ',origin_graph_index)
      #print('x at ',origin_graph_index,' is ', [node_ids_from_classes[ind] + '-> '+node_ids_from_classes[val.item()] for ind,val in enumerate(x.argmax(-1))])
      #print('y at ',origin_graph_index,' is ', [node_ids_from_classes[ind] +
                                        #'-> '+node_ids_from_classes[val.item()] for ind,val in enumerate(y_batch[origin_graph_index])])      
      #print('pred is ', [node_ids_from_classes[ind] + '-> '+node_ids_from_classes[val.item()] for ind,val in enumerate(pred.argmax(-1))])

      #print(x.argmax(-1))
      scene = torch.Tensor([0, 0, 0, 0, 2, 3, 3, 2, 10, 10, 0])  # coffee is on counter, what happends to cup
      temp_y = [int(x) for x in scene.tolist()]
      x_scene = y_to_x(temp_y)
      edge_scene = y_to_edge(temp_y)
      pred  = model(x_scene, edge_scene, y_index=graph_index)    
      #print('x at ',origin_graph_index,' is ', [node_ids_from_classes[ind] + '-> '+node_ids_from_classes[val.item()] for ind,val in enumerate(x_scene.argmax(-1))])      
      #print('pred is ', [node_ids_from_classes[ind] + '-> '+node_ids_from_classes[val.item()] for ind,val in enumerate(pred.argmax(-1))])

      #todo bfs
      bfs_search(2, train_data_list, model)