import torch
from torch_geometric.data import InMemoryDataset,Dataset, Data
import torch_geometric
import numpy as np
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #not show plots interactively
"""
0 - Kitchen
1 - Fridge
2 - Counter
3 - cabinet
4 - milk
5 - cereal
6 - coffee
7 - keys
8  -coffee cup
9 - cereal bowl
10 - utensil rack
 [0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]: this represents parent node of each node from 0 to 10, 
kitchen is child of  kitchen
fridge is child of  kitchen
counter is child of kitchen
cabinet is child of kitchen
milk is child of fridge
..
"""

def y_to_x(arr):
    # return one hot encoding as x given the target representation, y
    n = len(arr)
    x = np.zeros((n,n),dtype=int)
    for i in range(n):
        x[i][arr[i]] = 1
    return torch.Tensor(x )#, dtype=torch.float)

def y_to_edge(arr):
    # returns edge indexing given x
    #return y_to_fully_connected_edges(arr)
    index_start = []
    index_end = []
    n = len(arr)
    for i in range(n):
        if i!= arr[i]:
            index_start.append(i)
            index_end.append(arr[i])
    temp = index_start[:]
    index_start.extend(np.transpose(index_end)) #todo check if bidrectional edges
    index_end.extend(np.transpose(temp))
    op = []
    op.append(index_start)
    op.append(index_end)
    op = torch.Tensor(op)
    return op.long()

def y_to_fully_connected_edges(arr):
    n = len(arr)
    index_start = []
    index_end = []
    for i in range(n): #every nide connected to every node except itself
        for j in range(n):
            if i!=j:
                index_start.append(i)
                index_end.append(j)
    op = []
    temp = index_start[:]
    index_start.extend(np.transpose(index_end)) #todo check if bidrectional edges
    index_end.extend(np.transpose(temp))
    op.append(index_start)
    op.append(index_end)
    op = torch.Tensor(op)
    return op.long()
    

if __name__ == '__main__':

    # Making coffee - a
    #scene 1 - base state
    
    y_scene_1 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_1.tolist()]
    x_scene_1 = y_to_x(temp_y)
    edge_scene_1 = y_to_edge(temp_y)
    
    # scene 2 - coffee to counter
    y_scene_2 = torch.Tensor([0, 0, 0, 0, 1, 3, 2, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_2.tolist()]
    x_scene_2 = y_to_x(temp_y)
    edge_scene_2 = y_to_edge(temp_y)

    #scene 3 - Milk to counter
    y_scene_3 = torch.Tensor([0, 0, 0, 0, 2, 3, 2, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_3.tolist()]
    x_scene_3 = y_to_x(temp_y)
    edge_scene_3 = y_to_edge(temp_y)

    #scene 4 - Cup to counter
    y_scene_4 = torch.Tensor([0, 0, 0, 0, 2, 3, 2, 2, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_4.tolist()]
    x_scene_4 = y_to_x(temp_y)
    edge_scene_4 = y_to_edge(temp_y)

    #scene 5 - Coffee to cabinet
    y_scene_5 = torch.Tensor([0, 0, 0, 0, 2, 3, 3, 2, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_5.tolist()]
    x_scene_5 = y_to_x(temp_y)
    edge_scene_5 = y_to_edge(temp_y)

    #scene 6 - Milk to fridge
    y_scene_6 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_6.tolist()]
    x_scene_6 = y_to_x(temp_y)
    edge_scene_6 = y_to_edge(temp_y)

    #scene 7 - Cup to rack
    y_scene_7 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_7.tolist()]
    x_scene_7 = y_to_x(temp_y)
    edge_scene_7 = y_to_edge(temp_y)


    #a = Making coffee scenario 1
    Data_a_base_state = Data(x=x_scene_1, edge_index=edge_scene_1, y=y_scene_1)   # 5 steps
    Data_a_coffee_to_counter = Data(x=x_scene_1, edge_index=edge_scene_1, y=y_scene_2) # 1 step
    Data_a_milk_to_counter = Data(x=x_scene_2, edge_index=edge_scene_2, y=y_scene_3) # 1 step
    Data_a_cup_to_counter = Data(x=x_scene_3, edge_index=edge_scene_3,y=y_scene_4) # 1 step
    Data_a_making_coffee = Data(x=x_scene_4,edge_index=edge_scene_4,y=y_scene_4) # 5 steps
    Data_a_coffee_to_cabinet = Data(x=x_scene_4, edge_index=edge_scene_4, y=y_scene_5) # 1 step
    Data_a_milk_to_fridge = Data(x=x_scene_5, edge_index=edge_scene_5, y=y_scene_6) # 1 step 
    Data_a_cup_to_rack = Data(x=x_scene_6, edge_index=edge_scene_6, y=y_scene_7) # 1 step
    Data_a_end_state = Data(x=x_scene_7, edge_index=edge_scene_7, y=y_scene_7) # 4 steps
    
    num_nodes = Data_a_base_state.x.shape[0] #11
    ones = torch.ones(num_nodes)
    # we do not mask any nodes
    """
    Data_a_base_state.train_mask = ones.bool()
    Data_a_coffee_to_counter.train_mask = ones.bool()
    Data_a_milk_to_counter.train_mask = ones.bool()
    Data_a_cup_to_counter.train_mask = ones.bool()
    Data_a_making_coffee.train_mask = ones.bool()
    Data_a_coffee_to_cabinet.train_mask = ones.bool()
    Data_a_milk_to_fridge.train_mask = ones.bool()
    Data_a_cup_to_rack.train_mask = ones.bool()
    Data_a_end_state.train_mask = ones.bool()

    """
    # Has 20 time steps
    data_scenario_a_list = [                               
                            Data_a_base_state,
                            Data_a_coffee_to_counter,
                            Data_a_milk_to_counter,
                            Data_a_cup_to_counter,
                            Data_a_making_coffee, Data_a_making_coffee,
                            Data_a_making_coffee, Data_a_making_coffee,
                            Data_a_making_coffee,
                            Data_a_coffee_to_cabinet,
                            Data_a_milk_to_fridge,
                            Data_a_cup_to_rack,
                            Data_a_end_state]

    for data_object in data_scenario_a_list:
        data_object.train_mask = ones.bool()
        data_object.test_mask = ones.bool()
    #torch.save(data_scenario_a_list,'data_scenario_a.pt')
    
    ##############################################
    # Making cereal 
    #scene 8 - base state
    y_scene_8 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_8.tolist()]
    x_scene_8 = y_to_x(temp_y)
    edge_scene_8 = y_to_edge(temp_y)

    #scene 9 - Cereal to counter
    y_scene_9 = torch.Tensor([0, 0, 0, 0, 1, 2 ,3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_9.tolist()]
    x_scene_9 = y_to_x(temp_y)
    edge_scene_9 = y_to_edge(temp_y)

    #scene 10 - milk to counter
    y_scene_10 = torch.Tensor([0, 0, 0, 0, 2, 2, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_10.tolist()]
    x_scene_10 = y_to_x(temp_y)
    edge_scene_10 = y_to_edge(temp_y)

    #scene 11 - bowl to counter 
    y_scene_11 = torch.Tensor([0, 0, 0, 0, 2, 2, 3, 2, 10, 2, 0]) 
    temp_y = [int(x) for x in y_scene_11.tolist()]
    x_scene_11 = y_to_x(temp_y)
    edge_scene_11 = y_to_edge(temp_y)

    #scene 12 - cereal to cabinet 
    y_scene_12 = torch.Tensor([0, 0, 0, 0, 2, 3, 3, 2, 10, 2, 0]) 
    temp_y = [int(x) for x in y_scene_12.tolist()]
    x_scene_12 = y_to_x(temp_y)
    edge_scene_12 = y_to_edge(temp_y)

    #scene 13 - Milk to Fridge 
    y_scene_13 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 2, 0]) 
    temp_y = [int(x) for x in y_scene_13.tolist()]
    x_scene_13 = y_to_x(temp_y)
    edge_scene_13 = y_to_edge(temp_y)

    #scene 14 - Bowl to rack
    y_scene_14 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_14.tolist()]
    x_scene_14 = y_to_x(temp_y)
    edge_scene_14 = y_to_edge(temp_y)


    #b = Making cereal scenario 
    Data_b_base_state = Data(x=x_scene_8, edge_index=edge_scene_8, y=y_scene_8)   # 5 steps
    Data_b_cereal_to_counter = Data(x=x_scene_8, edge_index=edge_scene_8, y=y_scene_9) # 1 step
    Data_b_milk_to_counter = Data(x=x_scene_9, edge_index=edge_scene_9, y=y_scene_10) # 1 step
    Data_b_bowl_to_counter = Data(x=x_scene_10, edge_index=edge_scene_10,y=y_scene_11) # 1 step
    Data_b_making_cereal = Data(x=x_scene_11,edge_index=edge_scene_11,y=y_scene_11) # 5 steps
    Data_b_cereal_to_cabinet = Data(x=x_scene_11, edge_index=edge_scene_11, y=y_scene_12) # 1 step
    Data_b_milk_to_fridge = Data(x=x_scene_12, edge_index=edge_scene_12, y=y_scene_13) # 1 step 
    Data_b_bowl_to_rack = Data(x=x_scene_13, edge_index=edge_scene_13, y=y_scene_14) # 1 step
    Data_b_end_state = Data(x=x_scene_14, edge_index=edge_scene_14, y=y_scene_14) # 4 steps
    
    # Has 20 time steps
    data_scenario_b_list = [                              
                            Data_b_base_state,
                            Data_b_cereal_to_counter,
                            Data_b_milk_to_counter,
                            Data_b_bowl_to_counter,
                            Data_b_making_cereal, Data_b_making_cereal,
                            Data_b_making_cereal, Data_b_making_cereal,
                            Data_b_making_cereal,
                            Data_b_cereal_to_cabinet,
                            Data_b_milk_to_fridge,
                            Data_b_bowl_to_rack,
                            Data_b_end_state ]

    
    #torch.save(data_scenario_b_list,'data_scenario_b.pt')
    for data_object in data_scenario_b_list:
        data_object.train_mask = ones.bool()
        data_object.test_mask = ones.bool()


    #############################################
    # scenario c - Making coffee with stray object
    #scene 15 - base state
    y_scene_15 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_15.tolist()]
    x_scene_15 = y_to_x(temp_y)
    edge_scene_15 = y_to_edge(temp_y)

    #scene 16 - coffee to counter 
    y_scene_16 = torch.Tensor([0, 0, 0, 0, 1, 3, 2, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_16.tolist()]
    x_scene_16 = y_to_x(temp_y)
    edge_scene_16 = y_to_edge(temp_y)

    #scene 17 - milk to counter 
    y_scene_17 = torch.Tensor([0, 0, 0, 0, 2, 3, 2, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_17.tolist()]
    x_scene_17 = y_to_x(temp_y)
    edge_scene_17 = y_to_edge(temp_y)

    #scene 18 - cup to counter
    y_scene_18 = torch.Tensor([0, 0, 0, 0, 2, 3, 2, 2, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_18.tolist()]
    x_scene_18 = y_to_x(temp_y)
    edge_scene_18 = y_to_edge(temp_y)

    #scene 19 - key to kitchen
    y_scene_19 = torch.Tensor([0, 0, 0, 0, 2, 3, 2, 0, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_19.tolist()]
    x_scene_19 = y_to_x(temp_y)
    edge_scene_19 = y_to_edge(temp_y)

    #scene 20 - Key back to counter
    y_scene_20 = torch.Tensor([0, 0, 0, 0, 2, 3, 2, 2, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_20.tolist()]
    x_scene_20 = y_to_x(temp_y)
    edge_scene_20 = y_to_edge(temp_y)

    #scene 21 - Coffee to cabinet
    y_scene_21 = torch.Tensor([0, 0, 0, 0, 2, 3, 3, 2, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_21.tolist()]
    x_scene_21 = y_to_x(temp_y)
    edge_scene_21 = y_to_edge(temp_y)

    #scene 22 - Milk to Fridge
    y_scene_22 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_22.tolist()]
    x_scene_22 = y_to_x(temp_y)
    edge_scene_22 = y_to_edge(temp_y)

    #scene 23 - Cup to rack 
    y_scene_23 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_23.tolist()]
    x_scene_23 = y_to_x(temp_y)
    edge_scene_23 = y_to_edge(temp_y)

    Data_c_base_state = Data(x=x_scene_15, edge_index=edge_scene_15, y=y_scene_15)
    Data_c_coffee_to_counter = Data(x=x_scene_15, edge_index=edge_scene_15, y=y_scene_16)
    Data_c_milk_to_counter = Data(x=x_scene_16, edge_index=edge_scene_16, y=y_scene_17)
    Data_c_cup_to_counter = Data(x=x_scene_17, edge_index=edge_scene_17, y=y_scene_18)
    Data_c_key_to_kitchen = Data(x=x_scene_18, edge_index=edge_scene_18, y=y_scene_19) # when key falls down on kitchen floor from counter
    Data_c_key_to_counter = Data(x=x_scene_19, edge_index=edge_scene_19, y=y_scene_20)
    Data_c_making_coffee = Data(x=x_scene_20, edge_index=edge_scene_20, y=y_scene_20)
    Data_c_coffee_to_cabinet = Data(x=x_scene_20, edge_index=edge_scene_20, y=y_scene_21)
    Data_c_milk_to_fridge = Data(x=x_scene_21, edge_index=edge_scene_21, y=y_scene_22)
    Data_c_cup_to_rack = Data(x=x_scene_22, edge_index=edge_scene_22, y=y_scene_23)
    Data_c_end_state = Data(x=x_scene_23, edge_index=edge_scene_23, y=y_scene_23)

    data_scenario_c_list = [Data_c_base_state,
                            Data_c_coffee_to_counter,
                            Data_c_milk_to_counter,
                            Data_c_cup_to_counter,
                            Data_c_key_to_kitchen,
                            Data_c_key_to_counter,
                            Data_c_making_coffee,Data_c_making_coffee,
                            Data_c_making_coffee,Data_c_making_coffee,
                            Data_c_coffee_to_cabinet,
                            Data_c_milk_to_fridge,
                            Data_c_cup_to_rack,
                            ]
    
    #torch.save(data_scenario_c_list,'data_scenario_c.pt')
    for data_object in data_scenario_c_list:
        data_object.train_mask = ones.bool()
        data_object.test_mask = ones.bool()
    #################################################

    # scenario d - Making cereal with stray object

    #scene 24 Base state
    y_scene_24 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_24.tolist()]
    x_scene_24 = y_to_x(temp_y)
    edge_scene_24 = y_to_edge(temp_y)

    #scene 25 - Cereal to counter
    y_scene_25 = torch.Tensor([0, 0, 0, 0, 1, 2 ,3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_25.tolist()]
    x_scene_25 = y_to_x(temp_y)
    edge_scene_25 = y_to_edge(temp_y)

    #scene 26 - Milk to counter
    y_scene_26 = torch.Tensor([0, 0, 0, 0, 2, 2, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_26.tolist()]
    x_scene_26 = y_to_x(temp_y)
    edge_scene_26 = y_to_edge(temp_y)

    #scene 27 - Key to Kitchen
    y_scene_27 = torch.Tensor([0, 0, 0, 0, 2, 2, 3, 0, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_27.tolist()]
    x_scene_27 = y_to_x(temp_y)
    edge_scene_27 = y_to_edge(temp_y)

    #scene 28 - Key to Counter
    y_scene_28 = torch.Tensor([0, 0, 0, 0, 2, 2, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_28.tolist()]
    x_scene_28 = y_to_x(temp_y)
    edge_scene_28 = y_to_edge(temp_y)

    #scene 29 - Bowl to counter
    y_scene_29 = torch.Tensor([0, 0, 0, 0, 2, 2, 3, 2, 10, 2, 0]) 
    temp_y = [int(x) for x in y_scene_29.tolist()]
    x_scene_29 = y_to_x(temp_y)
    edge_scene_29 = y_to_edge(temp_y)

    #scene 30 - Cereal to cabinet
    y_scene_30 = torch.Tensor([0, 0, 0, 0, 2, 3, 3, 2, 10, 2, 0]) 
    temp_y = [int(x) for x in y_scene_30.tolist()]
    x_scene_30 = y_to_x(temp_y)
    edge_scene_30 = y_to_edge(temp_y)

    #scene 31 - Milk to fridge
    y_scene_31 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 2, 0]) 
    temp_y = [int(x) for x in y_scene_31.tolist()]
    x_scene_31 = y_to_x(temp_y)
    edge_scene_31 = y_to_edge(temp_y)

    #scene 32 - Bowl to rack
    y_scene_32 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_32.tolist()]
    x_scene_32 = y_to_x(temp_y)
    edge_scene_32 = y_to_edge(temp_y)

    Data_d_base_state = Data(x=x_scene_24, edge_index=edge_scene_24, y=y_scene_24)
    Data_d_cereal_to_counter = Data(x=x_scene_24, edge_index=edge_scene_24, y=y_scene_25)
    Data_d_milk_to_counter = Data(x=x_scene_25, edge_index=edge_scene_25, y=y_scene_26)
    Data_d_bowl_to_counter = Data(x=x_scene_26, edge_index=edge_scene_26, y=y_scene_27)
    Data_d_key_to_kitchen = Data(x=x_scene_27, edge_index=edge_scene_27, y=y_scene_28) # when key falls down on kitchen floor from counter
    Data_d_key_to_counter = Data(x=x_scene_28, edge_index=edge_scene_28, y=y_scene_29)
    Data_d_making_cereal = Data(x=x_scene_29, edge_index=edge_scene_29, y=y_scene_29)
    Data_d_cereal_to_cabinet = Data(x=x_scene_29, edge_index=edge_scene_29, y=y_scene_30)
    Data_d_milk_to_fridge = Data(x=x_scene_30, edge_index=edge_scene_30, y=y_scene_31)
    Data_d_bowl_to_rack = Data(x=x_scene_31, edge_index=edge_scene_31, y=y_scene_32)
    Data_d_end_state = Data(x=x_scene_32, edge_index=edge_scene_32, y=y_scene_32)

    data_scenario_d_list = [Data_d_base_state,
                            Data_d_milk_to_counter,
                            Data_d_bowl_to_counter,
                            Data_d_key_to_kitchen,
                            Data_d_key_to_counter,
                            Data_d_making_cereal,Data_d_making_cereal,
                            Data_d_making_cereal,Data_d_making_cereal,
                            Data_d_cereal_to_cabinet,
                            Data_d_milk_to_fridge,
                            Data_d_bowl_to_rack,
                            Data_d_end_state
                            ]
    
    #torch.save(data_scenario_d_list,'data_scenario_d.pt')
    for data_object in data_scenario_d_list:
        data_object.train_mask = ones.bool()
        data_object.test_mask = ones.bool()

    #####################################################

    #scenario e - Making coffee - 2nd type ( order of objects changed)
    #scene 33 - Base state
    y_scene_33 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_33.tolist()]
    x_scene_33 = y_to_x(temp_y)
    edge_scene_33 = y_to_edge(temp_y)

    #scene 34 - Milk to counter 
    y_scene_34 = torch.Tensor([0, 0, 0, 0, 2, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_34.tolist()]
    x_scene_34 = y_to_x(temp_y)
    edge_scene_34 = y_to_edge(temp_y)

    #scene 35 - Coffee to counter 
    y_scene_35 = torch.Tensor([0, 0, 0, 0, 2, 3, 2, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_35.tolist()]
    x_scene_35 = y_to_x(temp_y)
    edge_scene_35 = y_to_edge(temp_y)

    #scene 36 - Cup to counter
    y_scene_36 = torch.Tensor([0, 0, 0, 0, 2, 3, 2, 2, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_36.tolist()]
    x_scene_36 = y_to_x(temp_y)
    edge_scene_36 = y_to_edge(temp_y)

    #scene 37 - Milk to fridge
    y_scene_37 = torch.Tensor([0, 0, 0, 0, 1, 3, 2, 2, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_37.tolist()]
    x_scene_37 = y_to_x(temp_y)
    edge_scene_37 = y_to_edge(temp_y)

    #scene 38 - Coffee to cabinet
    y_scene_38 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 2, 10, 0]) 
    temp_y = [int(x) for x in y_scene_38.tolist()]
    x_scene_38 = y_to_x(temp_y)
    edge_scene_38 = y_to_edge(temp_y)

    #scene 39 - Cup to rack
    y_scene_39 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_39.tolist()]
    x_scene_39 = y_to_x(temp_y)
    edge_scene_39 = y_to_edge(temp_y)


    Data_e_base_state = Data(x=x_scene_33, edge_index=edge_scene_33, y=y_scene_33)
    Data_e_milk_to_counter = Data(x=x_scene_33, edge_index=edge_scene_33, y=y_scene_34)
    Data_e_coffee_to_counter = Data(x=x_scene_34, edge_index=edge_scene_34, y=y_scene_35)
    Data_e_cup_to_counter = Data(x=x_scene_35, edge_index=edge_scene_35, y=y_scene_36)
    Data_e_making_coffee = Data(x=x_scene_36, edge_index=edge_scene_36, y=y_scene_36)
    Data_e_milk_to_fridge = Data(x=x_scene_36, edge_index=edge_scene_36, y=y_scene_37)
    Data_e_coffee_to_cabinet = Data(x=x_scene_37, edge_index=edge_scene_37, y=y_scene_38)
    Data_e_cup_to_rack = Data(x=x_scene_38, edge_index=edge_scene_38, y=y_scene_39)
    Data_e_end_state = Data(x=x_scene_39, edge_index=edge_scene_39, y=y_scene_39) # same as initial 

    data_scenario_e_list = [ Data_e_base_state,
                            Data_e_milk_to_counter,
                            Data_e_coffee_to_counter,
                            Data_e_cup_to_counter,
                            Data_e_making_coffee,Data_e_making_coffee,
                            Data_e_making_coffee,Data_e_making_coffee,
                            Data_e_milk_to_fridge,
                            Data_e_coffee_to_cabinet,
                            Data_e_cup_to_rack,
                            Data_e_end_state, Data_e_end_state,
                            
                            ]
    
    #torch.save(data_scenario_e_list,'data_scenario_e.pt')
    for data_object in data_scenario_e_list:
        data_object.train_mask = ones.bool()
        data_object.test_mask = ones.bool()
    #############################################
    # scenario f - Making cereal V2

    #scene 40 - Base state
    y_scene_40 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_40.tolist()]
    x_scene_40 = y_to_x(temp_y)
    edge_scene_40 = y_to_edge(temp_y)

    #scene 41 - Milk to counter
    y_scene_41 = torch.Tensor([0, 0, 0, 0, 2, 3, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_41.tolist()]
    x_scene_41 = y_to_x(temp_y)
    edge_scene_41 = y_to_edge(temp_y)

    #scene 42 - Cereal to counter 
    y_scene_42 = torch.Tensor([0, 0, 0, 0, 2, 2, 3, 2, 10, 10, 0]) 
    temp_y = [int(x) for x in y_scene_42.tolist()]
    x_scene_42 = y_to_x(temp_y)
    edge_scene_42 = y_to_edge(temp_y)

    #scene 43 - Bowl to counter
    y_scene_43 = torch.Tensor([0, 0, 0, 0, 2, 2, 3, 2, 10, 2, 0]) 
    temp_y = [int(x) for x in y_scene_43.tolist()]
    x_scene_43 = y_to_x(temp_y)
    edge_scene_43 = y_to_edge(temp_y)

    #scene 44 - Milk to Fridge
    y_scene_44 = torch.Tensor([0, 0, 0, 0, 1, 2, 3, 2, 10, 2, 0]) 
    temp_y = [int(x) for x in y_scene_44.tolist()]
    x_scene_44 = y_to_x(temp_y)
    edge_scene_44 = y_to_edge(temp_y)

    #scene 45 - Cereal to cabinet
    y_scene_45 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 2, 0]) 
    temp_y = [int(x) for x in y_scene_45.tolist()]
    x_scene_45 = y_to_x(temp_y)
    edge_scene_45 = y_to_edge(temp_y)

    #scene 46 - Bowl to Rack 
    y_scene_46 = torch.Tensor([0, 0, 0, 0, 1, 3, 3, 2, 10, 10  , 0]) 
    temp_y = [int(x) for x in y_scene_46.tolist()]
    x_scene_46 = y_to_x(temp_y)
    edge_scene_46 = y_to_edge(temp_y)

    Data_f_base_state = Data(x=x_scene_40, edge_index=edge_scene_40, y=y_scene_40)
    Data_f_milk_to_counter = Data(x=x_scene_40, edge_index=edge_scene_40, y=y_scene_41)
    Data_f_cereal_to_counter = Data(x=x_scene_41, edge_index=edge_scene_41, y=y_scene_42)
    Data_f_bowl_to_counter = Data(x=x_scene_42, edge_index=edge_scene_42, y=y_scene_43)
    Data_f_making_cereal = Data(x=x_scene_43, edge_index=edge_scene_43, y=y_scene_43)
    Data_f_milk_to_fridge = Data(x=x_scene_43, edge_index=edge_scene_43, y=y_scene_44)
    Data_f_cereal_to_cabinet = Data(x=x_scene_44, edge_index=edge_scene_44, y=y_scene_45)
    Data_f_bowl_to_rack = Data(x=x_scene_45, edge_index=edge_scene_45, y=y_scene_46)
    Data_f_end_state = Data(x=x_scene_46, edge_index=edge_scene_46, y=y_scene_46)

    data_scenario_f_list = [Data_f_base_state, 
                            Data_f_milk_to_counter,
                            Data_f_cereal_to_counter,
                            Data_f_bowl_to_counter,
                            Data_f_making_cereal,Data_f_making_cereal,
                            Data_f_making_cereal,Data_f_making_cereal,
                            Data_f_milk_to_fridge,
                            Data_f_cereal_to_cabinet,
                            Data_f_bowl_to_rack,
                            Data_f_end_state, Data_f_end_state,
                            
                            ]
    
    #torch.save(data_scenario_f_list,'data_scenario_f.pt')
    for data_object in data_scenario_f_list:
        data_object.train_mask = ones.bool()
        data_object.test_mask = ones.bool()

    
    """
    A - 22 times 
    B - 20 times
    C - 4 times
    D - 4 times
    E - 20 times
    F - 20 times

    78  : A:18   B:16  C:3  D:3  E:16  F:16
    12  : A:4    B:4   C:1  D:1  E:4   F:4
    First transition. 
    check something from the first transition of a scenario.
    """
    train_data_list = []
   
    for i in range(18):
        train_data_list.extend(data_scenario_a_list)
    for i in range(16):
        train_data_list.extend(data_scenario_b_list)
    for i in range(3):
        train_data_list.extend(data_scenario_c_list)
    for i in range(3):
        train_data_list.extend(data_scenario_d_list)
    for i in range(16):
        train_data_list.extend(data_scenario_e_list)
    for i in range(16):
        train_data_list.extend(data_scenario_f_list)

    test_data_list = []
   
    for i in range(4):
        test_data_list.extend(data_scenario_a_list)
    for i in range(4):
        test_data_list.extend(data_scenario_b_list)
    for i in range(1):
        test_data_list.extend(data_scenario_c_list)
    for i in range(1):
        test_data_list.extend(data_scenario_d_list)
    for i in range(4):
        test_data_list.extend(data_scenario_e_list)
    for i in range(4):
        test_data_list.extend(data_scenario_f_list)

    # len(entire_data_list) 1800, 90 scenes* 20 length
    #print(len(train_data_list)) 1440
    #print(len(test_data_list)) 360

    torch.save(train_data_list, 'custom_data/train_custom_data_min_13_steps.pt')
    torch.save(test_data_list,'custom_data/test_custom_data_min_13_steps.pt')

    
    node_ids_from_classes = {0:'0: kitchen',1:'1: fridge',2:'2: counter',3:'3: cabinet',4:'4: milk',5:'5:cereal', 6:'coffee',
                             7:'7: keys', 8:'8: cup', 9:'9: bowl', 10:'10: rack'}

    g1 = torch_geometric.utils.to_networkx(Data_a_base_state)  #, to_undirected=True)
    g2 = torch_geometric.utils.to_networkx(Data_a_coffee_to_counter)
    g3 = torch_geometric.utils.to_networkx(Data_a_milk_to_counter)
    g4 = torch_geometric.utils.to_networkx(Data_a_cup_to_counter)
    f = plt.figure()
    nx.draw_networkx(g1, ax=f.add_subplot(221),labels=node_ids_from_classes)
    nx.draw_networkx(g2, ax=f.add_subplot(222),labels=node_ids_from_classes)
    nx.draw_networkx(g1, ax=f.add_subplot(223),labels=node_ids_from_classes)
    nx.draw_networkx(g2, ax=f.add_subplot(224),labels=node_ids_from_classes) #todo wrong output on figure
    
    f.savefig("base_graph.png")
