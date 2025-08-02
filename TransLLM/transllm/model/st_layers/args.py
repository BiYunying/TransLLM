import numpy as np
import pandas as pd
import configparser
import argparse
import os

curPath = os.path.abspath(os.path.dirname(__file__))
print('ST_Encoder', curPath)

def parse_args():
    # get configuration
    config_file = curPath + '/ST_Encoder.conf'
    config = configparser.ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser(prefix_chars='--', description='predictor_based_arguments')
    # data
    parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    parser.add_argument('--input_dim', type=int, default=config['data']['input_dim'])
    parser.add_argument('--output_dim', type=int, default=config['data']['output_dim'])
    parser.add_argument('--dataset', type=str, default=config['data']['dataset'])
    # model
    parser.add_argument('--time_stride', type=int, default=config['model']['time_stride'])
    # train
    parser.add_argument('--device', type=str, default=config['train']['device'])
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])
    parser.add_argument('--bs', type=int, default=config['train']['bs'])

    args, _ = parser.parse_known_args()
    return args