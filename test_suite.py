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
        
def generate_explanations(graph_node_indices, test_cases, top_5_feats_arr):
    
    node_ids_from_classes = {0:'kitchen',1:'fridge',2:'counter',3:'cabinet',4:'milk',5:'cereal', 6:'coffee',
                             7:'keys', 8:'cup', 9:'bowl', 10:'rack'}
    # ranking of test cases - higher score to same rankings and lower to rankings that are vastly different
    # rank most important from 5 to 1, get weighted average of intersection
    n = len(test_cases)
    len_feat_imp = len(top_5_feats_arr[0])
    weighted_imp = np.zeros(n)


    
    for i in range(n): # loop over test cases
        op = 0
        for j in range(len(test_cases[i])): # loop inside each test case                            
                if(test_cases[i][j] in top_5_feats_arr[i]):
                    temp_index = list(top_5_feats_arr[i]).index(test_cases[i][j])
                    op += (len_feat_imp - j)*(len_feat_imp - temp_index)                    
        weighted_imp[i] = op/len(test_cases[i])

    for i in range(len(test_cases)):
        print('Scenario {} : Object \'{}\', in the next step to be on {}.  Expected explanation nodes {},\n Got Top 5 explantions from Gnn as {}, Intersection {}, weighted importances {}'.format( 
              #graph_node_indices[i][0],  
              i+1,
              node_ids_from_classes[graph_node_indices[i][1]], node_ids_from_classes[graph_node_indices[i][2]], 
              [node_ids_from_classes[val] for val in test_cases[i]],
               [node_ids_from_classes[val] for val in top_5_feats_arr[i]], 
               [node_ids_from_classes[val] for val in  list(set( test_cases[i]) & set(top_5_feats_arr[i])) ] ,
               weighted_imp[i]
                )
              )


def scenario_testing(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    #graph_node_indices = [[5,6, 2], [6,4, 2],[7,8, 2],[9,6, 2],[9,4, 2]] # each sub list has values - graph index, node index, the target value of node
    graph_node_indices = [[1,6, 2], [2,4, 2],[3,8, 2],[5,6, 2],[5,4, 2], [10, 4,1], [11, 8, 10]] # each sub list has values - graph index, node index, the target value of node
    
    test_cases = [[2],[6,2],[6,4,2],[8,4,2],[6,4,2],[1,6], [10, 4, 6]]
    assert(len(graph_node_indices)==len(test_cases))

    weighted_outputs = []
    top_5_feats_arr = []

    for i in range(len(test_cases)):

        cfg['graph_index'] = graph_node_indices[i][0]
        cfg['node_index'] = graph_node_indices[i][1]
        expected_nodes = test_cases[i]
        top_5_feats, top_5_feat_imp = model_explanation_call(cfg)
        top_5_feats_arr.append(top_5_feats)
        weighted_op = 0
        for node in expected_nodes:
            ind = np.where(top_5_feats == node)
            if len(ind[0]):
                weighted_op += top_5_feat_imp[ind[0][0]]
        weighted_outputs.append(weighted_op)

    for i in range(len(test_cases)):
        print('Graph Index {}, Node index {}, Expected features {}, Top 5 features {}, Intersection {} and its importances {}'.format( 
              graph_node_indices[i][0],  graph_node_indices[i][1], test_cases[i], top_5_feats_arr[i], list(set( test_cases[i]) & set(top_5_feats_arr[i])) , weighted_outputs[i]))
    generate_explanations(graph_node_indices, test_cases, top_5_feats_arr)

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
   
    
     
        