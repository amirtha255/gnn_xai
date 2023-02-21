# gnn_xai
PyG explainer supports the following

algorithm:
GNNExplainer
CaptumExplainer - attribution_method='IntegratedGradients' , need node mask
PGExplainer - only supports phenomenon and 'none' node_mask_type


explanation_type:
"model": Explains the model prediction.
"phenomenon": Explains the phenomenon that the model is trying to predict.
compute their losses with respect to the model output ("model") or the target output ("phenomenon")

node_mask_type:
None: Will not apply any mask on nodes.
"object": Will mask each node.
"common_attributes": Will mask each feature.
"attributes": Will mask each feature across all nodes.

edge_mask_type:
Same options as node_mask_type

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

node_ids_from_classes = {0:'0: kitchen',1:'1: fridge',2:'2: counter',3:'3: cabinet',4:'4: milk',5:'5:cereal', 6:'coffee',
                                7:'7: keys', 8:'8: cup', 9:'9: bowl', 10:'10: rack'}
