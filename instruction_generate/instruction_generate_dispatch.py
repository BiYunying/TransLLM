import json
import pickle
import os
import numpy as np
import pandas as pd
from dataloader import get_dataloader_SH
import argparse
from dataloader import get_pretrain_task_batch_SH
import re

# =============================== Setting =============================== #
args = argparse.ArgumentParser(prefix_chars='--', description='test')
# NYCmulti(for train)     NYCtaxi NYCbike NYCcrime1 NYCcrime2 CHItaxi (for test)
args.add_argument('-dataset_name', default='SH_2015', type=str)
# args.add_argument('-dataset_name', default='SD_2021', type=str)
# Only one option can be set to True
args.add_argument('-for_zeroshot', default=False, type=eval, help='for zero-shot prediction or not')
args.add_argument('-for_supervised', default=False, type=eval, help='for supervised prediction or not')
args.add_argument('-for_ablation', default=False, type=eval, help='for ablation study or not')
args.add_argument('-for_test', default=True, type=eval, help='for test study or not')

args.add_argument('-his', default=12, type=int)
args.add_argument('-pre', default=1, type=int)
args.add_argument('-batch_size', default=1, type=int)
args.add_argument('-input_base_dim', default=1, type=int)
args.add_argument('-input_extra_dim', default=0, type=int)
args = args.parse_args()


if args.for_zeroshot:
    args.json_path = args.dataset_name + '_zeroshot.json'
    args.pkl_path = args.dataset_name + '_zeroshot_pkl.pkl'
elif args.for_supervised:
    args.json_path = args.dataset_name + '_supervised.json'
    args.pkl_path = args.dataset_name + '_supervised_pkl.pkl'
elif args.for_ablation:
    args.json_path = args.dataset_name + '_ablation.json'
    args.pkl_path = args.dataset_name + '_ablation_pkl.pkl'
elif args.for_test:
    args.json_path = args.dataset_name + '_test.json'
    args.pkl_path = args.dataset_name + '_test_pkl.pkl'
else:
    args.json_path = args.dataset_name + '.json'
    args.pkl_path = args.dataset_name + '_pkl.pkl'

if args.for_test:
    args.shuffle = False
else:
    args.shuffle = True
# =============================== Temporal Instructions =============================== #

time_ori_list_5m = []

for i in range(1, 289):
    hours = (i - 1) // 12
    minutes = (i - 1) % 12 * 5
    time_str = f"{hours:02d}:{minutes:02d}"
    time_ori_list_5m.append(time_str)
week_ori_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
month_ori_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                  'August', 'September', 'October', 'November', 'December']

def time_decode(data, args, type):
    month_start_index = int(data[:, 0, 0, args.input_base_dim + 5])
    day_start_index = int(data[:, 0, 0, args.input_base_dim + 4])
    year_start_index = int(data[:, 0, 0, args.input_base_dim + 6])
    time_start_index = int(data[:, 0, 0, args.input_base_dim + 2])
    week_start_index = int(data[:, 0, 0, args.input_base_dim + 3])


    month_start = month_ori_list[month_start_index-1]
    day_start= day_start_index
    year_start = year_start_index
    
    time_start = time_ori_list_5m[time_start_index - 1]
    week_start = week_ori_list[week_start_index-1]

    time_return = "'" + month_start + " " + str(day_start) + ", " + str(year_start) + ", " + \
                  time_start + ", " + week_start + "'"
    return time_return

# =============================== Spatial Instructions =============================== #
def region_decode_ori(region_idx, type, region_json_in):
    if type == 1:
        granularity = 'within a three-kilometer radius'
        region_idx = region_idx + 1
    else:
        granularity = 'within a one-kilometer radius'
    pois_categ_list = []
    region_index_info = region_json_in[str(region_idx)]
    if len(region_index_info) != 0:
        borough_name = region_index_info[0]['borough_name']
        for poi_index in region_index_info:
            pois_categ_list.append(poi_index['category_name'])
        pois_categ_list = list(set(pois_categ_list))
        pois_categ_str = str(pois_categ_list)[1:-1].replace("'", "")
        region_return = " This region is located within the " + borough_name + " borough district and " \
                         "encompasses various POIs " + granularity + ", covering " + pois_categ_str + \
                         " categories. "
    else:
        region_return = " No description is available for this region. "
    return region_return

def region_decode_others(region_idx, type, region_json_others):
    if type == 5:
        granularity = 'within a four-kilometer radius'
    else:
        granularity = 'within a one-kilometer radius'
    pois_categ_list = []
    region_index_info = region_json_others[region_idx]
    if len(region_index_info["name"]) != 0:
        city_name_list = region_index_info["vicinity"]
        for string in city_name_list:
            if ',' in string:
                after_comma = string.split(',', 1)[1].strip()
                city_name = after_comma
                break
        if 'city_name' not in locals():
            city_name = city_name_list[0]

        pois_categ_list = region_index_info["types"]
        pois_set = set(pois_categ_list)
        pois_set.discard('locality')
        pois_set.discard('point_of_interest')
        pois_categ_list = list(pois_set)[:10]
        pois_categ_str = str(pois_categ_list)[1:-1].replace('"', '').replace("'", "")
        region_return = " This region is located within the city of " + city_name + " and " \
                         "encompasses various POIs " + granularity + ", covering " + pois_categ_str + \
                         " categories. "
    else:
        region_return = " No description is available for this region. "
    return region_return


list_all = []
data_all = []


# =============================== data Generation =============================== #
x_trn, y_trn, mean, std, mean_waiting, std_waiting, real_prob = get_dataloader_SH(args)#生成X，Y
spt_x, spt_y,real_prob, train_len = get_pretrain_task_batch_SH(args, x_trn, y_trn,real_prob, shuffle=args.shuffle)# 将训练数据划分为批次并打乱
# mean, std =215.60181205171222,169.45704339035726
csv_file_path = "third_task_preprocess/square_grid_3km_shanghai.csv"
df = pd.read_csv(csv_file_path)


position_to_id = {}
for _, row in df.iterrows():
    position_to_id[(row['row'], row['col'])] = row['grid_id']

print(f"Created {len(position_to_id)} position indices")

result = []
valid_grids_data = []
valid_count = 0

# Define the relative positions of 9 directions (in order: center, top-left, top, top-right, left, right, bottom-left, bottom, bottom-right)
directions = [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

print("Start filtering grids...")
for idx, row in df.iterrows():
    grid_row = row['row']
    grid_col = row['col']
    
    # Check whether all 9 grids in the 3x3 neighborhood exist and collect their grid_ids
    all_neighbors_exist = True
    nine_direction_ids = []
    
    for dr, dc in directions:
        neighbor_pos = (grid_row + dr, grid_col + dc)
        if neighbor_pos in position_to_id:
            nine_direction_ids.append(position_to_id[neighbor_pos]-1)
        else:
            all_neighbors_exist = False
            break
    
    if all_neighbors_exist:
        result.append(1)
        valid_grids_data.append(nine_direction_ids)
        valid_count += 1
    else:
        result.append(0)

nine_direction_array = np.array(valid_grids_data) if valid_grids_data else np.array([]).reshape(0, 9)

print(f"\nFiltering completed!")
print(f"Total number of grids: {len(df)}")
print(f"Number of valid grids: {valid_count}")
print(f"Filtering ratio:  {valid_count/len(df)*100:.1f}%")
print(f"Shape of 9-direction array: {nine_direction_array.shape}")
region_starts = set()
for i in range(train_len):
    data, label, prob = spt_x[i], spt_y[i],real_prob[i]
    print(i, train_len)

    data_x_nor = data[..., :args.input_base_dim]

    data_nor = (data_x_nor - mean) / std

    data_x_nor_waiting = data[..., 2:3]

    data_nor_waiting = (data_x_nor_waiting - mean_waiting) / std_waiting


    dict_data = {}

    time_of_day = data[..., 3:4]/288
    day_of_week = (data[..., 4:5]-1)/7
    dict_data["data_x"], dict_data["data_y"] = np.concatenate([data_nor[..., :args.input_base_dim],time_of_day,day_of_week, data[..., -1:]], axis=-1), \
                                               np.concatenate([label[..., :args.input_base_dim], label[..., -1:]], axis=-1)
    dict_data["real_prob"] = prob
    dict_data["mean"] = mean
    dict_data["std"] = std
    dict_data["data_x_waiting"], dict_data["data_y_waiting"] = np.concatenate([data_nor_waiting[..., 0:1],time_of_day,day_of_week, data[..., -1:]], axis=-1), \
                                               np.concatenate([label[..., 2:3], label[..., -2:3]], axis=-1)
     
    dict_data["mean_waiting"] = mean_waiting
    dict_data["std_waiting"] = std_waiting
    data_all.append(dict_data)


    region_nums = data.shape[2]
    data = data[:,-1:,:,:]
    tmp_idx = -1
    for region_index in range(0, data.shape[2], 1):
        if result[region_index] == 1:
            tmp_idx+=1
            if data[:, :, region_index, 1]>10:
                region_start = region_index
                region_end = region_index + 1
                if region_end > (region_nums - 1):
                    region_end = region_nums
                list_conversations = []
                dict_main = {}
                dict_conversation_human = {}
                dict_conversation_gpt = {}

                list_gpt_demand = []
                list_gpt_empty = []

                
                neighbor_indices = nine_direction_array[tmp_idx].astype(int)
                data_neighbors = data[:, :, neighbor_indices, :]
                
                for dim_index in range(args.input_base_dim):
                    list_gpt_demand.append(data_neighbors[0, :, :, dim_index][0].astype(int))
                    list_gpt_empty.append(data_neighbors[0, :, :, 1][0].astype(int))

                # =============================== Format Standardization =============================== #

                str_empty = str(list_gpt_empty[0]).replace(",", "")
                str_empty = re.sub(r'\s+', ' ', str_empty)
                if str_empty[1] == " ":
                    str_empty = str_empty[:1] + str_empty[2:]

                # =============================== Instruction Generated =============================== #
                type = data[0, 0, 0, -1]

                region_index_new = region_index 

                if type ==9:
                    # value_of_human = "Your task is to redistribute empty taxis from the center grid (Grid 0) to nearby regions " \
                    #         "to maximize future passenger-taxi matching and successful ride completion rates, while minimizing unnecessary travel. " \
                    #         "The grid is divided into a 3x3 region centered around the current grid (Grid 0). Each grid is indexed as follows: [0: center, 1: top-left, 2: top, 3: top-right, 4: left, 5: right, 6: bottom-left, 7: bottom, 8: bottom-right]." \
                    #         " Current empty taxi count per grid: "+ str_empty  +\
                    #         ". Current demand count per grid: "+ str_demand  +\
                    #         ". The current time is " +time_decode(data_neighbors, args, type)+", and the dispatch planning is for the period until " +time_decode(label, args, type)+ \
                    #         ". We use a pre-trained spatiotemporal encoder to represent the predicted demand of the 3x3 region at the next time step as <ST_EMB>. " \
                    #         "You must decide the probability distribution for dispatching taxis from Grid 0 to each destination (including staying in Grid 0). " \
                    #         "Consider demand-supply imbalance, proximity preferences, and integrate both current patterns and future predictions <ST_EMB> to optimize expected matching while minimizing travel costs." \
                    #         " Output a 9-dimensional probability vector summing to 1.0 representing dispatch probabilities for [stay, top-left, top, top-right, left, right, bottom-left, bottom, bottom-right], and express it as <ST_PRE>."
                    value_of_human = "Your task is to redistribute empty taxis from the center grid (Grid 0) to nearby regions " \
                            "to maximize future passenger-taxi matching and successful ride completion rates, while minimizing unnecessary travel. " \
                            "The grid is divided into a 3x3 region centered around the current grid (Grid 0). Each grid is indexed as follows: [0: center, 1: top-left, 2: top, 3: top-right, 4: left, 5: right, 6: bottom-left, 7: bottom, 8: bottom-right]." \
                            " Current empty taxi count per grid: "+ str_empty  +\
                            ". The current time is " +time_decode(data_neighbors, args, type)+", and the dispatch planning is for the period until " +time_decode(label, args, type)+ \
                            ". We use a pre-trained spatiotemporal encoder to represent the predicted demand and the number of idle vehicles in the 3*3 region during the dispatching period, denoted as <ST_EMB> and <ST_EMB>. " \
                            "You must decide the probability distribution for dispatching taxis from Grid 0 to each destination (including staying in Grid 0). " \
                            "Consider demand-supply imbalance, proximity preferences, and integrate both current patterns and future predictions embeddings to optimize expected matching while minimizing travel costs." \
                            " Output a 9-dimensional probability vector summing to 1.0 representing dispatch probabilities for [stay, top-left, top, top-right, left, right, bottom-left, bottom, bottom-right], and express it as <ST_PRE>."
                
                    value_of_gpt = "Based on the given information, the recommended dispatch probabilities for taxis in this region are <ST_PRE>."
                dict_main["id"] = 'train_' + args.dataset_name + '_region_' + str(region_start) + '_' + str(region_end) + '_len_' + str(i)
                dict_conversation_human["from"], dict_conversation_human["value"] = "human", value_of_human
                dict_conversation_gpt["from"], dict_conversation_gpt["value"] = "gpt", value_of_gpt
                list_conversations.append(dict_conversation_human)
                list_conversations.append(dict_conversation_gpt)
                dict_main["conversations"] = list_conversations
                list_all.append(dict_main)
                region_starts.add(region_index)
                  
#You are an intelligent traffic dispatch agent. Your task is to redistribute empty taxis from the center grid (Grid 0) to nearby regions to maximize future passenger-taxi matching and successful ride completion rates, while minimizing unnecessary travel. The grid is divided into a 3x3 region centered around the current grid (Grid 0). Each grid is indexed as follows: [0: center, 1: top-left, 2: top, 3: top-right, 4: left, 5: right, 6: bottom-left, 7: bottom, 8: bottom-right]. Current empty taxi count per grid: [12, 5, 7, 3, 6, 4, 2, 3, 1]. Current demand count per grid: [8, 2, 6, 1, 4, 3, 1, 2, 1]. We use a pre-trained spatiotemporal encoder to represent the predicted demand of the 3x3 region at the next time step as <ST_EMB>. You must decide the probability distribution for dispatching taxis from Grid 0 to each destination (including staying in Grid 0). Consider demand-supply imbalance, proximity preferences, and integrate both current patterns and future predictions <ST_EMB> to optimize expected matching while minimizing travel costs. Output a 9-dimensional probability vector summing to 1.0 representing dispatch probabilities for [stay, top-left, top, top-right, left, right, bottom-left, bottom, bottom-right].            
for i in region_starts:
    if result[i] == 0:
        print(i)
# =============================== .json and .pkl Saved =============================== #
folder_path = './data/prompt_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' was created.")
json_savepath = os.path.join(folder_path, args.json_path)
b = json.dumps(list_all)
f2 = open(json_savepath, 'w')
f2.write(b)
b=None
f2.close()
pkl_savepath = os.path.join(folder_path, args.pkl_path)
with open(pkl_savepath, 'wb') as file:
    pickle.dump(data_all, file)