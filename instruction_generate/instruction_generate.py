import json
import pickle
import os
import numpy as np

from dataloader import get_dataloader
import argparse
from dataloader import get_pretrain_task_batch
import re

# =============================== Setting =============================== #
args = argparse.ArgumentParser(prefix_chars='--', description='test')
# NYCmulti(for train)     NYCtaxi NYCbike NYCcrime1 NYCcrime2 CHItaxi (for test)
# args.add_argument('-dataset_name', default='SH_2015', type=str)
# args.add_argument('-dataset_name', default='SD_2021', type=str)
args.add_argument('-dataset_name', default='pems04', type=str)
# Only one option can be set to True
args.add_argument('-for_zeroshot', default=True, type=eval, help='for zero-shot prediction or not')
args.add_argument('-for_supervised', default=False, type=eval, help='for supervised prediction or not')
args.add_argument('-for_ablation', default=False, type=eval, help='for ablation study or not')
args.add_argument('-for_test', default=False, type=eval, help='for test study or not')

args.add_argument('-his', default=12, type=int)
args.add_argument('-pre', default=12, type=int)
args.add_argument('-batch_size', default=1, type=int)
args.add_argument('-input_base_dim', default=1, type=int)
args.add_argument('-input_extra_dim', default=0, type=int)
args.add_argument('-part_of_region', default=False, type=eval)
args.add_argument('-region_start', default=0, type=int)
args.add_argument('-region_end', default=80, type=int)
# args.add_argument('-region_end', default=1, type=int)
args = args.parse_args()

if args.dataset_name == 'NYCmulti':
    args.for_test = False
    args.json_path = args.dataset_name + '.json'
    args.pkl_path = args.dataset_name + '_pkl.pkl'
else:
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
time_ori_list = []
time_ori_list_5m = []
time_ori_list_60m = []
for i in range(1, 49):
    hours = (i - 1) // 2
    minutes = (i - 1) % 2 * 30
    time_str = f"{hours:02d}:{minutes:02d}"
    time_ori_list.append(time_str)
for i in range(1, 25):
    hours = (i - 1) * 1
    minutes = 0
    time_str = f"{hours:02d}:{minutes:02d}"
    time_ori_list_60m.append(time_str)
for i in range(1, 289):
    hours = (i - 1) // 12
    minutes = (i - 1) % 12 * 5
    time_str = f"{hours:02d}:{minutes:02d}"
    time_ori_list_5m.append(time_str)
week_ori_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
month_ori_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                  'August', 'September', 'October', 'November', 'December']

def time_decode(data, args, type):
    month_start_index = int(data[:, 0, 0, args.input_base_dim + 4])
    month_end_index = int(data[:, -1, 0, args.input_base_dim + 4])
    day_start_index = int(data[:, 0, 0, args.input_base_dim + 3])
    day_end_index = int(data[:, -1, 0, args.input_base_dim + 3])
    year_start_index = int(data[:, 0, 0, args.input_base_dim + 5])
    year_end_index = int(data[:, -1, 0, args.input_base_dim + 5])
    time_start_index = int(data[:, 0, 0, args.input_base_dim + 1])
    time_end_index = int(data[:, -1, 0, args.input_base_dim + 1])
    week_start_index = int(data[:, 0, 0, args.input_base_dim + 2])
    week_end_index = int(data[:, -1, 0, args.input_base_dim + 2])

    month_start, month_end = month_ori_list[month_start_index-1], month_ori_list[month_end_index-1]
    day_start, day_end = day_start_index, day_end_index
    year_start, year_end = year_start_index, year_end_index
    if type == 3 or type == 4 or type == 6 or type==11:
        time_start, time_end = time_ori_list_60m[time_start_index - 1], time_ori_list_60m[time_end_index - 1]
    elif type == 1 or type == 2 or type == 5:
        time_start, time_end = time_ori_list[time_start_index-1], time_ori_list[time_end_index-1]
    else:
        time_start, time_end = time_ori_list_5m[time_start_index - 1], time_ori_list_5m[time_end_index - 1]
    week_start, week_end = week_ori_list[week_start_index-1], week_ori_list[week_end_index-1]

    if type == 3 or type == 4 or type == 6 or type==11:
        interval = str(60) + "-minutes intervals'"
    elif type == 1 or type == 2 or type == 5:
        interval = str(30) + "-minute intervals'"
    else:
        interval = str(5) + "-minute intervals'"
    # interval_str = "prediction points" if isPred else "data points"
    interval_str = "data points"
    time_return = "'" + month_start + " " + str(day_start) + ", " + str(year_start) + ", " + \
                  time_start + ", " + week_start + " to " + month_end + " " + str(day_end) + ", " + \
                  str(year_end) + ", " + time_end + ", " + week_end + ", with " + interval_str + \
                  " recorded at " + interval
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
x_trn,x_1d,x_1w, y_trn, mean, std = get_dataloader(args)#生成X，Y
spt_x,spt_xd,spt_xw, spt_y, train_len = get_pretrain_task_batch(args, x_trn,x_1d,x_1w, y_trn, shuffle=args.shuffle)# 将训练数据划分为批次并打乱
# mean, std =215.60181205171222,169.45704339035726
for i in range(train_len):
    data,data_1d,data_1w, label = spt_x[i],spt_xd[i],spt_xw[i], spt_y[i]
    print(i, train_len)
    # generate st_data_all
    if args.part_of_region:
        data = data[:, :, args.region_start:args.region_end, :]
        data_1d = data_1d[:, :, args.region_start:args.region_end, :]
        data_1w = data_1w[:, :, args.region_start:args.region_end, :]
        label = label[:, :, args.region_start:args.region_end, :]

    data_x_nor = data[..., :args.input_base_dim]
    data1d_x_nor = data_1d[..., :args.input_base_dim]
    data1w_x_nor = data_1w[..., :args.input_base_dim]
    
    data_nor = (data_x_nor - mean) / std
    data1d_nor = (data1d_x_nor - mean) / std
    data1w_nor = (data1w_x_nor - mean) / std

    dict_data = {}

    # time_of_day = data[..., 2:3]/288
    # day_of_week = (data[..., 3:4]-1)/7
    time_of_day = data[..., 2:3]/24
    day_of_week = (data[..., 3:4]-1)/7
    time_of_day_1d = data_1d[..., 2:3]/24
    day_of_week_1d = (data_1d[..., 3:4]-1)/7
    time_of_day_1w = data_1w[..., 2:3]/24
    day_of_week_1w = (data_1w[..., 3:4]-1)/7
    dict_data["data_x"], dict_data["data_y"] = np.concatenate([data_nor[..., :args.input_base_dim],time_of_day,day_of_week, data[..., -1:]], axis=-1), \
                                               np.concatenate([label[..., :args.input_base_dim], label[..., -1:]], axis=-1)
    
    # time_of_day_1d = data_1d[..., 2:3]/288
    # day_of_week_1d = (data_1d[..., 3:4]-1)/7
    # time_of_day_1w = data_1w[..., 2:3]/288
    # day_of_week_1w = (data_1w[..., 3:4]-1)/7
    dict_data["data_x_1d"], dict_data["data_x_1w"] = np.concatenate([data1d_nor[..., :args.input_base_dim],time_of_day_1d,day_of_week_1d, data1d_nor[..., -1:]], axis=-1), \
                                               np.concatenate([data1w_nor[..., :args.input_base_dim],time_of_day_1w,day_of_week_1w, data1w_nor[..., -1:]], axis=-1)
    
    dict_data["mean"] = mean
    dict_data["std"] = std
    data_all.append(dict_data)

    region_nums = data.shape[2]
    for region_index in range(0, data.shape[2], 1):
        region_start = region_index
        region_end = region_index + 1
        if region_end > (region_nums - 1):
            region_end = region_nums
        list_conversations = []
        dict_main = {}
        dict_conversation_human = {}
        dict_conversation_gpt = {}

        list_gpt_datain = []
        list_gpt_lblsin = []
        data_gpt = data[:, :, region_start:region_end, :]
        label_gpt = label[:, :, region_start:region_end, :]
        for dim_index in range(args.input_base_dim):
            list_gpt_datain.append(data_gpt[0, :, 0, dim_index].astype(int))
            list_gpt_lblsin.append(label_gpt[0, :, 0, dim_index].astype(int))

        # =============================== Format Standardization =============================== #
        str_flow = str(list_gpt_datain[0]).replace(",", "")
        str_flow = re.sub(r'\s+', ' ', str_flow)
        if str_flow[1] == " ":
            str_flow = str_flow[:1] + str_flow[2:]
        if args.input_base_dim > 1:
            str_outflow = str(list_gpt_datain[1]).replace(",", "")
            str_outflow = re.sub(r'\s+', ' ', str_outflow)
            if str_outflow[1] == " ":
                str_outflow = str_outflow[:1] + str_outflow[2:]

        lbls_inflow = str(list_gpt_lblsin[0]).replace(",", "")
        lbls_inflow = re.sub(r'\s+', ' ', lbls_inflow)
        if lbls_inflow[1] == " ":
            lbls_inflow = lbls_inflow[:1] + lbls_inflow[2:]


        # =============================== Instruction Generated =============================== #
        type = data_gpt[0, 0, 0, -1]

        region_index_new = region_index + args.region_start

        # value_of_human = "Given the historical data for traffic flow over 12 time steps at a highway traffic monitoring point, " \
        #          "the recorded traffic flow values are " + str_flow + \
        #          ". The recording time of the historical data is " + time_decode(data_gpt, args, type) + \
        #          ". To capture spatial and temporal dependencies, a spatio-temporal convolution model is utilized to " \
        #          "encode the historical traffic data as embeddings <ST_EMB>. Additionally, time encoding features such as " \
        #          "hour, minute, and day of the week are incorporated as <TIME_ENC>. " \
        #          "Now we want to predict the traffic flow for the next 12 time steps during the time period of " + \
        #          time_decode(label_gpt, args, type) + ". Please analyze the traffic patterns in this region, taking into " \
        #          "account the provided historical data, time encoding, and spatio-temporal embeddings, and generate the " \
        #          "predictive tokens for regression in the form \"<ST_PRE>\"."
        if type ==7 or type ==10.0:
            value_of_human = "Given the historical data for traffic flow over 12 time steps at a highway traffic monitoring point, " \
                    "the recorded traffic flow values are " + str_flow + \
                    ". The recording time of the historical data is " + time_decode(data_gpt, args, type) + \
                    ". To capture spatial and temporal dependencies, a spatio-temporal convolution model is utilized to " \
                    "encode the historical traffic data as embeddings <ST_EMB>. " \
                    "Now we want to predict the traffic flow for the next 12 time steps during the time period of " + \
                    time_decode(label_gpt, args, type) + ". Please analyze the traffic patterns in this region, taking into " \
                    "account the provided historical data, time encoding, and spatio-temporal embeddings, and generate the " \
                    "predictive tokens for regression in the form \"<ST_PRE>\"."
            value_of_gpt = "Based on the given historical traffic flow data, time encoding, and spatio-temporal embeddings, the " \
                        "predictive tokens for the traffic flow in this region are <ST_PRE>."
        elif type ==8 or type ==11.0:
            value_of_human = "Given the historical data for charging demand over 12 time steps in a grid of Shenzhen, " \
                    "the recorded charging demand values are " + str_flow + \
                    ". The recording time of the historical data is " + time_decode(data_gpt, args, type) + \
                    ". To capture spatial and temporal dependencies, a spatio-temporal convolution model is utilized to " \
                    "encode the historical data as embeddings <ST_EMB>. " \
                    "Now we want to predict the charging demand for the next 12 time steps during the time period of " + \
                    time_decode(label_gpt, args, type) + ". Please analyze the charging patterns in this region, taking into " \
                    "account the provided historical data, time encoding, and spatio-temporal embeddings, and generate the " \
                    "predictive tokens for regression in the form \"<ST_PRE>\"."
            value_of_gpt = "Based on the given charging demand data, time encoding, and spatio-temporal embeddings, the " \
                        "predictive tokens for the charging demand in this region are <ST_PRE>."
        dict_main["id"] = 'train_' + args.dataset_name + '_region_' + str(region_start) + '_' + str(region_end) + '_len_' + str(i)
        dict_conversation_human["from"], dict_conversation_human["value"] = "human", value_of_human
        dict_conversation_gpt["from"], dict_conversation_gpt["value"] = "gpt", value_of_gpt
        list_conversations.append(dict_conversation_human)
        list_conversations.append(dict_conversation_gpt)
        dict_main["conversations"] = list_conversations
        list_all.append(dict_main)
        
            

# =============================== .json and .pkl Saved =============================== #
folder_path = './data/prompt_data1'
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