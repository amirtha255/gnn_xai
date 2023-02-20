
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
from torch_geometric.explain import Explainer, GNNExplainer , PGExplainer, CaptumExplainer

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

PATH = "trained_simple_model.pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

#set
algorithm = GNNExplainer(epochs=200)
algorithm = PGExplainer(epochs=200) # only phenomenon and none type node mask
algorithm = CaptumExplainer(attribution_method='IntegratedGradients') # need node mask
"""
CaptumExplainer
PGExplainer
"""
explanation_type='phenomenon'
"""
"model": Explains the model prediction.
"phenomenon": Explains the phenomenon that the model is trying to predict.
compute their losses with respect to the model output ("model") or the target output ("phenomenon")
"""
node_mask_type='attributes'
"""
None: Will not apply any mask on nodes.
"object": Will mask each node.
"common_attributes": Will mask each feature.
"attributes": Will mask each feature across all nodes.
"""
edge_mask_type='object'
"""
Same options as node_mask_type
"""
model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    )
"""
mode (ModelMode or str) –
The mode of the model. The possible values are:
"binary_classification": A binary classification model.
"multiclass_classification": A multiclass classification model.
"regression": A regression model.

task_level (ModelTaskLevel or str) –
The task-level of the model. The possible values are:
"node": A node-level prediction model.
"edge": An edge-level prediction model.
"graph": A graph-level prediction model.

return_type (ModelReturnType or str, optional) –
The return type of the model. The possible values are (default: None):
"raw": The model returns raw values.
"probs": The model returns probabilities.
"log_probs": The model returns log-probabilities.
"""

explainer = Explainer(
    model=model,
    algorithm=algorithm,
    explanation_type=explanation_type,
    node_mask_type=node_mask_type,
    edge_mask_type=edge_mask_type,
    model_config=model_config,
)

node_ids_from_classes = {0:'0: kitchen',1:'1: fridge',2:'2: counter',3:'3: cabinet',4:'4: milk',5:'5:cereal', 6:'coffee',
                             7:'7: keys', 8:'8: cup', 9:'9: bowl', 10:'10: rack'}


graph_index =10
node_index = 2
target = (train_data_list[graph_index].y).long().to(device)
x, edge_index = train_data_list[graph_index].x.to(device),train_data_list[graph_index].edge_index.to(device) #choose which graph to explain

explanation = explainer(x, edge_index, index=node_index, target=target)
print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=5)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path)
print(f"Subgraph visualization plot has been saved to '{path}'")

