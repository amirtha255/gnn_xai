import os
import yaml
import numpy as np
from run_explainer import simple_model_explanations
# amirtha todo, same params yield different explanations in different runs, why ??

# can use test suite - one file for different kinds of scenarios testing
# or test same scenario with multiple model configs.
def model_testing(cfg_folder):
    test_case = [2,4] 
    config_files = []
    weighted_outputs = []
    
    for filename in os.listdir(cfg_folder):
        file = os.path.join(cfg_folder,filename)
        config_files.append(file)

    for i in range(len(config_files)):
        with open(config_files[i]) as f:
            cfg = yaml.safe_load(f)
        top_5_feats, top_5_feat_imp = simple_model_explanations(cfg)
        weighted_op = 0
        for node in test_case:
            ind = np.where(top_5_feats == node)
            if len(ind[0]):
                weighted_op += top_5_feat_imp[ind[0][0]]
        weighted_outputs.append(weighted_op)
        print('cfg file, and its weighted importances', config_files[i], weighted_outputs[i])
        

def scenario_testing(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    graph_node_indices = [[0,0], [2,2],[7,4],[8,6],[14,4]]
    test_cases = [[2,7],[2,9],[2,4],[2],[2]]
    assert(len(graph_node_indices)==len(test_cases))

    weighted_outputs = []

    for i in range(len(test_cases)):

        cfg['graph_index'] = graph_node_indices[i][0]
        cfg['node_index'] = graph_node_indices[i][1]
        expected_nodes = test_cases[i]
        top_5_feats, top_5_feat_imp = simple_model_explanations(cfg)
        weighted_op = 0
        for node in expected_nodes:
            ind = np.where(top_5_feats == node)
            if len(ind[0]):
                weighted_op += top_5_feat_imp[ind[0][0]]
        weighted_outputs.append(weighted_op)
        print('Expected op, and its weighted importances', test_cases[i], weighted_outputs[i])

if __name__ == '__main__':

    config_path = 'xai_config/default.yaml'
    config_folder = 'xai_config'
    mode = 'scenario_testing'
    #mode='model_testing'

    if mode == 'scenario_testing':
        scenario_testing(config_path)
    elif mode == 'model_testing':
        model_testing(config_folder)
    else:
        print('mode is not valid')
   
    
     
        