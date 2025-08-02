import torch
import numpy as np
import torch.utils.data
from add_window import Add_Window_Horizon,Add_Window_Horizon_SH
from load_dataset import load_st_dataset
from normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import random


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
    data = load_st_dataset(args.dataset_name, args)        # B, N, D
    data = data.transpose(1, 0, 2)
    
    data_x_nor = data[..., :1]

    mean = np.mean(data_x_nor) 
    std = np.std(data_x_nor)

    x_tra,x_1d,x_1w, y_tra = Add_Window_Horizon(data, args.his, args.pre, single)
   
    print('Train: ', x_tra.shape, y_tra.shape)
    return x_tra,x_1d,x_1w, y_tra, mean, std
def get_dataloader_SH(args, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
    #load raw st dataset
    data,real_prob = load_st_dataset(args.dataset_name, args)        # B, N, D
    data = data.transpose(1, 0, 2)
    real_prob = real_prob.transpose(1, 0, 2)
    data_x_nor = data[..., :1]

    data_x_nor_waiting = data[..., 2:3]
    
    mean = np.mean(data_x_nor)
    std = np.std(data_x_nor)

    mean_waiting= np.mean(data_x_nor_waiting) 
    std_waiting = np.std(data_x_nor_waiting)
   
    x_tra, y_tra, real_prob = Add_Window_Horizon_SH(data,real_prob, args.his, args.pre, single)
   
    print('Train: ', x_tra.shape, y_tra.shape)
    return x_tra, y_tra, mean, std,mean_waiting,std_waiting,real_prob
def get_pretrain_task_batch(args, x_tra,x_1d,x_1w, y_tra, shuffle=True):
    batch_size = args.batch_size
    len_dataset = x_tra.shape[0]

    batch_list_x = []
    batch_list_xd = []
    batch_list_xw = []
    batch_list_y = []
    permutation = np.random.permutation(len_dataset)
    for index in range(0, len_dataset, batch_size):
        start = index
        end = min(index + batch_size, len_dataset)
        indices = permutation[start:end]
        if shuffle:
            x_data = x_tra[indices.copy()]
            xd_data = x_1d[indices.copy()]
            xw_data = x_1w[indices.copy()]
            y_data = y_tra[indices.copy()]
        else:
            x_data = x_tra[start:end]
            xd_data = x_1d[start:end]
            xw_data = x_1w[start:end]
            y_data = y_tra[start:end]
        batch_list_x.append(x_data)
        batch_list_xd.append(xd_data)
        batch_list_xw.append(xw_data)
        batch_list_y.append(y_data)
    train_len = len(batch_list_x)
    return batch_list_x,batch_list_xd,batch_list_xw, batch_list_y, train_len
def get_pretrain_task_batch_SH(args, x_tra, y_tra,real_prob, shuffle=True):
    batch_size = args.batch_size
    len_dataset = x_tra.shape[0]

    batch_list_x = []
    batch_list_y = []
    batch_list_prob = []
    permutation = np.random.permutation(len_dataset)
    for index in range(0, len_dataset, batch_size):
        start = index
        end = min(index + batch_size, len_dataset)
        indices = permutation[start:end]
        if shuffle:
            x_data = x_tra[indices.copy()]
            y_data = y_tra[indices.copy()]
            prob_data = real_prob[indices.copy()]
        else:
            x_data = x_tra[start:end]
            y_data = y_tra[start:end]
            prob_data = real_prob[start:end]
        batch_list_x.append(x_data)
        batch_list_y.append(y_data)
        batch_list_prob.append(prob_data)
    train_len = len(batch_list_x)
    return batch_list_x, batch_list_y,batch_list_prob, train_len


def normalize_dataset(data, normalizer, input_base_dim, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            scaler = StandardScaler(mean, std)
            data[:, :, 0:input_base_dim] = scaler.transform(data[:, :, 0:input_base_dim])
        else:
            data_ori = data[:, :, 0:input_base_dim]
            data_day = data[:, :, input_base_dim:input_base_dim+1]
            data_week = data[:, :, input_base_dim+1:input_base_dim+2]

            mean_data = data_ori.mean()
            std_data = data_ori.std()
            mean_day = data_day.mean()
            std_day = data_day.std()
            mean_week = data_week.mean()
            std_week = data_week.std()

            scaler_data = StandardScaler(mean_data, std_data)
            data_ori = scaler_data.transform(data_ori)
            scaler_day = StandardScaler(mean_day, std_day)
            data_day = scaler_day.transform(data_day)
            scaler_week = StandardScaler(mean_week, std_week)
            data_week = scaler_week.transform(data_week)
            data = np.concatenate([data_ori, data_day, data_week], axis=-1)
            print(mean_data, std_data, mean_day, std_day, mean_week, std_week)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler_data, scaler_day, scaler_week, None

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data