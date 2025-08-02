import numpy as np


def Add_Window_Horizon(data, window=12, horizon=12, single=False):
    '''
    :param data: shape [T, ...], one data point every 5 minutes  
    :param window: window size in data points (default 12 = 1 hour)  
    :param horizon: prediction horizon  
    :param single: whether to predict only a single time point  
    :return:  
        X_current, X_1h, X_2h, X_1d, X_1w: [B, window, ...]  
        Y: [B, horizon, ...] or [B, 1, ...]  '''

    points_per_hour = 12
    offset_1h = 12 * 1      # 12
    offset_1d = points_per_hour * 24     # 288
    offset_1w = points_per_hour * 24 * 7 # 2016

    length = len(data)
    min_required_offset = offset_1w  
    end_index = length - horizon - window + 1

    X_1h, X_1d, X_1w = [], [], []
    Y = []

    for idx in range(min_required_offset, end_index):
        if (idx - offset_1h < 0  or 
            idx - offset_1d < 0 or idx - offset_1w < 0):
            continue

        if (idx + window + horizon > length):
            continue

        X_1h.append(data[idx - offset_1h: idx - offset_1h + window])
        X_1d.append(data[idx - offset_1d: idx - offset_1d + window])
        X_1w.append(data[idx - offset_1w: idx - offset_1w + window])

        if single:
            Y.append(data[idx + horizon - 1: idx + horizon])
        else:
            Y.append(data[idx: idx + window])

    return  np.array(X_1h), np.array(X_1d), np.array(X_1w), np.array(Y)

def Add_Window_Horizon_SH(data,real_prob, window=12, horizon=1, single=False):
    '''
    :param data: shape [T, ...], one data point every 5 minutes
    :param window: window size in data points (default 12 = 1 hour)
    :param horizon: prediction horizon
    :param single: whether to predict only a single time point
    :return:
        X_current, X_1h, X_2h, X_1d, X_1w: [B, window, ...]
        Y: [B, horizon, ...] or [B, 1, ...]
    '''


    points_per_hour = 12
    offset_1h = points_per_hour * 1      # 12

    length = len(data)

    end_index = length - horizon 

    X_1h = []
    Y = []
    real_prob_list =[]
    for idx in range(end_index):
        if (idx - offset_1h < 0 ):
            continue
        
        X_1h.append(data[idx - offset_1h: idx - offset_1h + window])
        real_prob_list.append(real_prob[idx : idx +1])

        if single:
            Y.append(data[idx + horizon - 1: idx + horizon])
        else:
            Y.append(data[idx: idx + horizon])

    return  np.array(X_1h), np.array(Y),np.array(real_prob_list)