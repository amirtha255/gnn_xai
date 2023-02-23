import os
import yaml
import numpy as np
from run_explainer import simple_model_explanations
from run_explainer_spatio_custom_model import spatio_explanations
# todo, same params yield different explanations in different runs, why ??
#assumptions - node features are used, that is how we get summation of feature importances in the explainer

# can use test suite - one file for different kinds of scenarios testing, or test same scenario with multiple model configs.
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
        top_5_feats, top_5_feat_imp = model_explanation_call(cfg)
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
    graph_node_indices = [[5,6], [6,4],[7,8],[9,6],[9,4]]
    test_cases = [[2],[2,6],[2,4,6],[2,4,8],[2,4,6]]
    assert(len(graph_node_indices)==len(test_cases))

    weighted_outputs = []

    for i in range(len(test_cases)):

        cfg['graph_index'] = graph_node_indices[i][0]
        cfg['node_index'] = graph_node_indices[i][1]
        expected_nodes = test_cases[i]
        top_5_feats, top_5_feat_imp = model_explanation_call(cfg)
        weighted_op = 0
        for node in expected_nodes:
            ind = np.where(top_5_feats == node)
            if len(ind[0]):
                weighted_op += top_5_feat_imp[ind[0][0]]
        weighted_outputs.append(weighted_op)
        print('Expected op, and its weighted importances', test_cases[i], weighted_outputs[i])

if __name__ == '__main__':

    #todo set here
    mode = 'scenario_testing'     #mode='model_testing'
    model = 'spatio'    # model = 'spatio'

    if model=='simple':
        config_path = 'xai_config/default.yaml'
        config_folder = 'xai_config'
        model_explanation_call = simple_model_explanations
    elif model=='spatio':
        config_path = 'spatio_xai_config/xai_spatio_custom.yaml'
        config_folder = 'spatio_xai_config'
        model_explanation_call = spatio_explanations
    else:
        print('model is not valid')
        

    if mode == 'scenario_testing':
        scenario_testing(config_path)
    elif mode == 'model_testing': #test all config files which correspond to different models inside config folder
        model_testing(config_folder)
    else:
        print('mode is not valid')
   
    
     
        