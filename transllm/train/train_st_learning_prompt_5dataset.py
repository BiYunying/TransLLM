# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re
from dataclasses import dataclass, field
import json
import logging
import pathlib
import argparse
import pickle
from typing import Dict, Optional, Sequence, List
import pandas as pd
from typing import Any
import torch
import numpy as np
from transllm.model.utils import preprocess
print(f"Available devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
import transformers
from torch.utils.data import Dataset
from TransLLM.transllm.train.stchat_trainer import STChatTrainer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from transllm import conversation as conversation_lib
from transllm.model import *
from transllm.model.STLlama_learning_prompt_5dataset import STLlamaForCausalLM

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_STHIS_TOKEN = "<ST_EMB>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_st_mlp_adapter: bool = field(default=False)
    st_tower: Optional[str] = field(default=None)
    st_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_st_mlp_adapter: Optional[str] = field(default=None)
    use_st_start_end: bool = field(default=False)
    num_prompts: int = 0
    num_slots: int = 0

@dataclass
class DataArguments:
    data_path_sd: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_path_pems08: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_path_sz: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_path_urbanev: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_path_sh: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_st: bool = False
    sep_st_conv_front: bool = False
    st_token_len: int = 0
    st_content: Optional[str] = field(default=None)
    st_data_path_sd: Optional[str] = field(default=None)
    st_data_path_pems08: Optional[str] = field(default=None)
    st_data_path_sz: Optional[str] = field(default=None)
    st_data_path_urbanev: Optional[str] = field(default=None)
    st_data_path_sh: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_st_mlp_adapter: bool = field(default=False)
    freeze_prompt_router: bool = field(default=True)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.01
    lora_weight_path: str = ""
    lora_bias: str = "none"
    disable_tqdm: bool = False

def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
         'CA': [base_dir+'st_data/ca', base_dir+'st_data/ca/ca_rn_adj.npy', 8600],
         'GLA': [base_dir+'st_data/gla', base_dir+'st_data/gla/gla_rn_adj.npy', 3834],
         'GBA': [base_dir+'st_data/gba', base_dir+'st_data/gba/gba_rn_adj.npy', 2352],
         'SD': [base_dir+'st_data/sd', base_dir+'st_data/sd/sd_rn_adj.npy', 673],
         'shenzhen':[base_dir+'st_data/shenzhen', base_dir+'st_data/shenzhen/shenzhen_adj.npy', 247],
         'shanghai':[base_dir+'st_data/shanghai', base_dir+'st_data/shanghai/shanghai_adj.npy', 891],
         'urbanev':[base_dir+'st_data/urbanev', base_dir+'st_data/urbanev/urbanev_adj.npy', 275],
         'pems08':[base_dir+'st_data/pems08', base_dir+'st_data/pems08/pems08_adj.npy', 170],
         'pems03':[base_dir+'st_data/pems03', base_dir+'st_data/pems03/pems03_adj_clip.npy', 170],
         'pems04':[base_dir+'st_data/pems04', base_dir+'st_data/pems04/pems04_adj_clip.npy', 170]
        }
    assert dataset in d.keys()
    return d[dataset]

def read_node_information(args):
    file_path = args.node_information_path1
    df = pd.read_csv(file_path)
    selected_cols = ["Fwy", "Lanes", "Direction"]
    df = df[selected_cols].copy()
    df["Fwy_main"] = df["Fwy"].str.split("-").str[0]
    # 2.1 Fwy：one hot
    encoder_fwy = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    fwy_encoded = encoder_fwy.fit_transform(df[["Fwy_main"]])

    # 2.2 Lanes:
    lanes_raw = df[["Lanes"]].values.astype(int)
    valid_directions = ["N", "S", "W", "E"]

    encoder_direction = OneHotEncoder(
        categories=[valid_directions],
        sparse_output=False,
        handle_unknown="ignore"
    )
    direction_encoded = encoder_direction.fit_transform(df[["Direction"]])

    final_features = np.hstack([fwy_encoded, lanes_raw, direction_encoded])
    return final_features

def read_node_information2(args):
    file_path = args.node_information_path2
    df = pd.read_csv(file_path)
    
    selected_cols = ["count", "fast_count", "slow_count"]
    df = df[selected_cols].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.values)  # shape: [N, 3]

    final_features = np.hstack([scaled_features])
    return final_features

def read_node_information4(args):
    file_path = args.node_information_path4
    df = pd.read_csv(file_path)
    
    selected_cols = ["charge_count"] 
    df = df[selected_cols].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.values)  # shape: [N, 3]

    final_features = np.hstack([scaled_features])
    return final_features

def load_adj(args):
    nodes_feature1 = read_node_information(args)
    nodes_feature1 = torch.from_numpy(nodes_feature1).float().to(args.device)
    data_path, adj_path, node_num = get_dataset_info(args.dataset1)
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    sp_matrix = adj_mx + np.transpose(adj_mx)
    rows, cols = np.where(sp_matrix)
    sp_matrix = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    batch_offsets = torch.arange(args.bs, device=args.device) * node_num
    sp_matrix = sp_matrix.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    sp_matrix = sp_matrix.reshape(2,-1)


    nodes_feature2 = read_node_information2(args)
    nodes_feature2 = torch.from_numpy(nodes_feature2).float().to(args.device)
    se_matrix =np.load(os.path.join(data_path, "cached_dist_matrix.npy"))
    rows, cols = np.where(se_matrix)
    se_matrix = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix = se_matrix.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix = se_matrix.reshape(2,-1)

    data_path2, adj_path2, node_num2 = get_dataset_info(args.dataset2)
    sp_matrix2 = load_adj_from_numpy(adj_path2)
    rows, cols = np.where(sp_matrix2)
    sp_matrix2 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    batch_offsets = torch.arange(args.bs, device=args.device) * node_num2
    sp_matrix2 = sp_matrix2.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    sp_matrix2 = sp_matrix2.reshape(2,-1)

    se_matrix2 =np.load(os.path.join(data_path2, "cached_dist_matrix.npy"))
    rows, cols = np.where(se_matrix2)
    se_matrix2 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix2 = se_matrix2.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix2 = se_matrix2.reshape(2,-1)

    data_path3, adj_path3, node_num3 = get_dataset_info(args.dataset3)
    sp_matrix3 = load_adj_from_numpy(adj_path3)
    rows, cols = np.where(sp_matrix3)
    sp_matrix3 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    batch_offsets = torch.arange(args.bs, device=args.device) * node_num3
    sp_matrix3 = sp_matrix3.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    sp_matrix3 = sp_matrix3.reshape(2,-1)

    se_matrix3 =np.load(os.path.join(data_path3, "cached_demand_matrix.npy"))
    rows, cols = np.where(se_matrix3)
    se_matrix3 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix3 = se_matrix3.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix3 = se_matrix3.reshape(2,-1)

    se_matrix3_2 =np.load(os.path.join(data_path3, "cached_waiting_matrix.npy"))
    rows, cols = np.where(se_matrix3_2) 
    se_matrix3_2 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix3_2 = se_matrix3_2.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix3_2 = se_matrix3_2.reshape(2,-1)
    
    nodes_feature4 = read_node_information4(args)
    nodes_feature4 = torch.from_numpy(nodes_feature4).float().to(args.device)
    data_path4, adj_path4, node_num4 = get_dataset_info(args.dataset4)
    sp_matrix4 = load_adj_from_numpy(adj_path4)
    rows, cols = np.where(sp_matrix4)
    sp_matrix4 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    batch_offsets = torch.arange(args.bs, device=args.device) * node_num4
    sp_matrix4 = sp_matrix4.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    sp_matrix4 = sp_matrix4.reshape(2,-1)

    se_matrix4 =np.load(os.path.join(data_path4, "cached_dist_matrix.npy"))
    rows, cols = np.where(se_matrix4)
    se_matrix4 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix4 = se_matrix4.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix4 = se_matrix4.reshape(2,-1)

    data_path5, adj_path5, node_num5 = get_dataset_info(args.dataset5)
    sp_matrix5 = load_adj_from_numpy(adj_path5)
    rows, cols = np.where(sp_matrix5)
    sp_matrix5 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    batch_offsets = torch.arange(args.bs, device=args.device) * node_num5
    sp_matrix5 = sp_matrix5.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    sp_matrix5 = sp_matrix5.reshape(2,-1)

    se_matrix5 =np.load(os.path.join(data_path5, "cached_dist_matrix.npy"))
    rows, cols = np.where(se_matrix5)
    se_matrix5 = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
    se_matrix5 = se_matrix5.unsqueeze(2) + batch_offsets.view(1, 1, -1)
    se_matrix5 = se_matrix5.reshape(2,-1)

    return nodes_feature1,sp_matrix,se_matrix,nodes_feature2,sp_matrix2,se_matrix2,sp_matrix3,se_matrix3,se_matrix3_2,nodes_feature4,sp_matrix4,se_matrix4,sp_matrix5,se_matrix5

def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            module_name = names[0] if len(names) == 1 else names[-1]
            name1 = names[1] if len(names) != 1 else names[-1]
            if 'st_pred_linear' not in module_name and 'st_projector' not in module_name and not name.startswith('prompt_router') and 'st_tower' not in name1:
                lora_module_names.add(name)
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset1', type=str, default='SD')
    parser.add_argument('--dataset2', type=str, default='shenzhen')
    parser.add_argument('--dataset3', type=str, default='shanghai')
    parser.add_argument('--dataset4', type=str, default='urbanev')
    parser.add_argument('--dataset5', type=str, default='pems08')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='2021')
    parser.add_argument('--model_name', type=str, default='localgat_3dataset')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--tpd', type=int, default=288, help='time per day')
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--thres', type=float, default=0.6)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--time_stride', type=int, default=1)
    parser.add_argument('--node_information_path1', type=str, default='data/st_data/sd/sd_meta.csv')
    parser.add_argument('--node_information_path2', type=str, default='data/st_data/shenzhen/sz_meta.csv')
    parser.add_argument('--node_information_path4', type=str, default='data/st_data/urbanev/meta.csv')
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args() 
    return args




class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset_ST(Dataset):
    def __init__(self, data_path_sd: str, data_path_sz: str,data_path_sh: str, data_path_pems08: str, data_path_urbanev: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 st_cfg: dict,
                 batch_size: int = 16,
                 **kwargs):
        super(LazySupervisedDataset_ST, self).__init__()
        logging.warning("Loading data...")
        self.sd_list_data_dict = json.load(open(data_path_sd, "r"))
        self.pems08_list_data_dict = json.load(open(data_path_pems08, "r"))
        self.sz_list_data_dict = json.load(open(data_path_sz, "r"))
        self.urbanev_list_data_dict = json.load(open(data_path_urbanev, "r"))
        self.sh_list_data_dict = json.load(open(data_path_sh, "r"))

        self.tokenizer = tokenizer
        self.st_cfg = st_cfg
        self.st_data_all_sd = pickle.load(open(kwargs.get('st_data_path_sd'), 'rb'))
        self.st_data_all_pems08 = pickle.load(open(kwargs.get('st_data_path_pems08'), 'rb'))
        self.st_data_all_sz = pickle.load(open(kwargs.get('st_data_path_sz'), 'rb'))
        self.st_data_all_urbanev = pickle.load(open(kwargs.get('st_data_path_urbanev'), 'rb'))
        self.st_data_all_sh = pickle.load(open(kwargs.get('st_data_path_sh'), 'rb'))
        args = get_config()
        args.bs = batch_size
        self.nodes_feature1,self.sp_matrix1,self.se_matrix1,\
            self.nodes_feature2,self.sp_matrix2,self.se_matrix2,\
                self.sp_matrix3,self.se_matrix3,self.se_matrix4,self.nodes_feature5,\
                    self.sp_matrix5,self.se_matrix5,self.sp_matrix6,self.se_matrix6=load_adj(args)

        csv_file_path = "third_task_preprocess/square_grid_3km_shanghai.csv"
        df = pd.read_csv(csv_file_path)
        position_to_id = {}
        for _, row in df.iterrows():
            position_to_id[(row['row'], row['col'])] = row['grid_id']
        result = []
        valid_grids_data = []  
        valid_count = 0

        # Define the relative positions for 9 directions (in order: center, top-left, top, top-right, left, right, bottom-left, bottom, bottom-right)
        directions = [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for idx, row in df.iterrows():
            grid_row = row['row']
            grid_col = row['col']
            
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

        self.nine_direction_array = np.array(valid_grids_data) if valid_grids_data else np.array([]).reshape(0, 9)
        self.batch_size = batch_size

        self.build_index_sequence()

    def build_index_sequence(self):
    # """Construct index sequences by batch, where each batch of 16 samples comes from the same dataset"""
        import random
        sd_indices = list(range(len(self.sd_list_data_dict)))
        pems08_indices = list(range(len(self.pems08_list_data_dict)))
        sz_indices = list(range(len(self.sz_list_data_dict)))
        urbanev_indices = list(range(len(self.urbanev_list_data_dict)))
        sh_indices = list(range(len(self.sh_list_data_dict)))
        random.shuffle(sd_indices)
        random.shuffle(pems08_indices)
        random.shuffle(sz_indices)
        random.shuffle(urbanev_indices)
        random.shuffle(sh_indices)

        def split_batches(indices):
            return [indices[i:i+self.batch_size]
                    for i in range(0, len(indices), self.batch_size)
                    if len(indices[i:i+self.batch_size]) == self.batch_size]

        sd_batches = split_batches(sd_indices)
        pems08_batches = split_batches(pems08_indices)
        sz_batches = split_batches(sz_indices)
        urbanev_batches = split_batches(urbanev_indices)
        sh_batches = split_batches(sh_indices)

        self.index_sequence = []
        sd_i, sz_i, sh_i, pems08_i, urbanev_i = 0, 0, 0, 0, 0

        while sd_i < len(sd_batches):
            # SD batch
            for _ in range(1):
                if sd_i < len(sd_batches):
                    for idx in sd_batches[sd_i]:
                        self.index_sequence.append(('SD', idx))
                    sd_i += 1
                else:
                    break

            #  pems08 batch
            if len(pems08_batches) > 0:
                for idx in pems08_batches[pems08_i % len(pems08_batches)]:
                    self.index_sequence.append(('pems08', idx))
                pems08_i += 1


            # batch
            if len(sz_batches) > 0:
                for idx in sz_batches[sz_i % len(sz_batches)]:
                    self.index_sequence.append(('SZ', idx))
                sz_i += 1

            # urbanev batch
            if len(urbanev_batches) > 0:
                for idx in urbanev_batches[urbanev_i % len(urbanev_batches)]:
                    self.index_sequence.append(('urbanev', idx))
                urbanev_i += 1


            # SH batch
            if len(sh_batches) > 0:
                for idx in sh_batches[sh_i % len(sh_batches)]:
                    self.index_sequence.append(('SH', idx))
                sh_i += 1

        
    def __len__(self):
        return len(self.index_sequence)

    def __getitem__(self, i):
        dataset_type, real_idx = self.index_sequence[i]
        if dataset_type == 'SD':
            sources = self.sd_list_data_dict[real_idx]
            st_data_all = self.st_data_all_sd
            nodes_feature = self.nodes_feature1
            sp_matrix = self.sp_matrix1
            se_matrix = self.se_matrix1
            se_matrix_waiting = None
        elif dataset_type == 'SZ':
            sources = self.sz_list_data_dict[real_idx]
            st_data_all = self.st_data_all_sz
            nodes_feature = self.nodes_feature2
            sp_matrix = self.sp_matrix2
            se_matrix = self.se_matrix2
            se_matrix_waiting = None
        elif dataset_type == 'SH':
            sources = self.sh_list_data_dict[real_idx]
            st_data_all = self.st_data_all_sh
            nodes_feature = None
            sp_matrix = self.sp_matrix3
            se_matrix = self.se_matrix3
            se_matrix_waiting = self.se_matrix4
            neighbors = self.nine_direction_array
        elif dataset_type == 'pems08':
            sources = self.pems08_list_data_dict[real_idx]
            st_data_all = self.st_data_all_pems08
            nodes_feature = None
            sp_matrix = self.sp_matrix6
            se_matrix = self.se_matrix6
            se_matrix_waiting = None
        elif dataset_type == 'urbanev':
            sources = self.urbanev_list_data_dict[real_idx]
            st_data_all = self.st_data_all_urbanev
            nodes_feature = self.nodes_feature5
            sp_matrix = self.sp_matrix5
            se_matrix = self.se_matrix5
            se_matrix_waiting = None
        if dataset_type == 'pems08' or dataset_type == 'urbanev':
            region_start = int(sources["id"].split('_')[3])
            region_end = int(sources["id"].split('_')[4])
            i4data_all = int(sources["id"].split('_')[6])
        else:
            region_start = int(sources["id"].split('_')[4])
            region_end = int(sources["id"].split('_')[5])
            i4data_all = int(sources["id"].split('_')[7])
        data_dict = dict()
        data_dict['st_data_x'] = torch.tensor(st_data_all[i4data_all]['data_x'], dtype=torch.float32)
        data_dict['st_data_y'] = torch.tensor(st_data_all[i4data_all]['data_y'], dtype=torch.float32)
        data_dict['mean'] = torch.tensor(st_data_all[i4data_all]['mean'], dtype=torch.float32)
        data_dict['std'] = torch.tensor(st_data_all[i4data_all]['std'], dtype=torch.float32)
        if dataset_type == 'SH':
            data_dict['st_data_x_waiting'] = torch.tensor(st_data_all[i4data_all]['data_x_waiting'], dtype=torch.float32)
            data_dict['st_data_y_waiting'] = torch.tensor(st_data_all[i4data_all]['data_y_waiting'], dtype=torch.float32)
            data_dict['mean_waiting'] = torch.tensor(st_data_all[i4data_all]['mean_waiting'], dtype=torch.float32)
            data_dict['std_waiting'] = torch.tensor(st_data_all[i4data_all]['std_waiting'], dtype=torch.float32)
            data_dict['real_prob'] = torch.tensor(st_data_all[i4data_all]['real_prob'], dtype=torch.float32)
        else:
            data_dict['st_data_x_waiting'] = None
            data_dict['st_data_y_waiting'] = None
            data_dict['mean_waiting'] = None
            data_dict['std_waiting'] = None
            data_dict['real_prob'] = None
        data_dict['region_start'] = region_start
        data_dict['region_end'] = region_end
        data_dict['sources'] = [sources]
        data_dict['sp_matrix'] = torch.tensor(sp_matrix)
        data_dict['se_matrix'] = torch.tensor(se_matrix)
        if dataset_type == 'SD' or dataset_type == 'SZ' or dataset_type == 'urbanev':
            data_dict['st_data_xd'] = torch.tensor(st_data_all[i4data_all]['data_x_1d'], dtype=torch.float32)
            data_dict['st_data_xw'] = torch.tensor(st_data_all[i4data_all]['data_x_1w'], dtype=torch.float32)
            data_dict['nodes_feature'] = torch.tensor(nodes_feature, dtype=torch.float32)
            data_dict['neighbors'] = None
            data_dict['se_matrix_waiting'] = None
        elif dataset_type == 'pems08':
            data_dict['st_data_xd'] = torch.tensor(st_data_all[i4data_all]['data_x_1d'], dtype=torch.float32)
            data_dict['st_data_xw'] = torch.tensor(st_data_all[i4data_all]['data_x_1w'], dtype=torch.float32)
            data_dict['nodes_feature'] = None
            data_dict['neighbors'] = None
            data_dict['se_matrix_waiting'] = None
        elif dataset_type == 'SH':
            data_dict['st_data_xd'] = None
            data_dict['st_data_xw'] = None
            data_dict['nodes_feature'] = None
            data_dict['neighbors'] = torch.tensor(neighbors, dtype=torch.float32)
            data_dict['se_matrix_waiting'] = torch.tensor(se_matrix_waiting)
        return data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        batch = dict(
            sources=[ins["sources"] for ins in instances],
            st_data_x=[ins["st_data_x"] for ins in instances],
            st_data_xd=[ins["st_data_xd"] for ins in instances],
            st_data_xw=[ins["st_data_xw"] for ins in instances],
            st_data_y=[ins["st_data_y"] for ins in instances],
            mean=[ins["mean"] for ins in instances],
            std=[ins["std"] for ins in instances],
            st_data_x_waiting=[ins["st_data_x_waiting"] for ins in instances],
            st_data_y_waiting=[ins["st_data_y_waiting"] for ins in instances],
            mean_waiting=[ins["mean_waiting"] for ins in instances],
            std_waiting=[ins["std_waiting"] for ins in instances],
            region_start=[ins["region_start"] for ins in instances],
            region_end=[ins["region_end"] for ins in instances],
            nodes_feature=[ins["nodes_feature"] for ins in instances],
            sp_matrix=[ins["sp_matrix"] for ins in instances],
            se_matrix=[ins["se_matrix"] for ins in instances],
            se_matrix_waiting=[ins["se_matrix_waiting"] for ins in instances],
            neighbors=[ins["neighbors"] for ins in instances],
            real_prob=[ins["real_prob"] for ins in instances],
        )
        return batch


def make_supervised_stdata_module(tokenizer: transformers.PreTrainedTokenizer,
                                  data_args,batch_size) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    print('lazy_preprocess', data_args.lazy_preprocess)
    dataset_cls = (LazySupervisedDataset_ST
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path_sd=data_args.data_path_sd,
                                data_path_sz=data_args.data_path_sz,
                                data_path_sh=data_args.data_path_sh,
                                data_path_pems08=data_args.data_path_pems08,
                                data_path_urbanev=data_args.data_path_urbanev,
                                st_cfg=dict(
                                    is_st=data_args.is_st,
                                    sep_st_conv_front=data_args.sep_st_conv_front,
                                    st_token_len=data_args.st_token_len,
                                    st_content=data_args.st_content,
                                    use_st_start_end=getattr(data_args, 'use_st_start_end', False)
                                ),
                                batch_size= batch_size,
                                st_data_path_sd=data_args.st_data_path_sd,
                                st_data_path_sz=data_args.st_data_path_sz,
                                st_data_path_sh=data_args.st_data_path_sh,
                                st_data_path_pems08=data_args.st_data_path_pems08,
                                st_data_path_urbanev=data_args.st_data_path_urbanev)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(model_args, data_args, training_args):
    
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

    model_args, data_args, training_args = model_args, data_args, training_args   
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    print('compute_dtype', compute_dtype)

    bnb_model_from_pretrained_args = {}

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.st_tower is not None:
        model = STLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_prompts = model_args.num_prompts,
            num_slots = model_args.num_slots,
            **bnb_model_from_pretrained_args
        )  ## TODO: add real ST Llama model
        model.set_tokenizer(tokenizer)
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    print('model_args.version: ', model_args.version)
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    elif model_args.version == "v2":
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3_v2_1"]
        tokenizer.pad_token = None
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
    else:
        tokenizer.pad_token = tokenizer.unk_token #<unk>
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]
    
    print(model.config.pretrain_ST_model_path)
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    
    if training_args.gradient_checkpointing and model_args.st_tower is None:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            print('require_grads: input')
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            print('require_grads: output')

    if training_args.lora_enable:
        print('lora_enable:', training_args.lora_enable)
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        logging.warning("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if model_args.st_tower is not None:
        model_st_dict = model.get_model().initialize_st_modules(
            st_tower=model_args.st_tower,
            st_select_layer=model_args.st_select_layer,
            pretrain_st_mlp_adapter=model_args.pretrain_st_mlp_adapter,
            fsdp=training_args.fsdp
        )
        model.get_st_tower().to(dtype=torch.float32, device=training_args.device)

        data_args.is_st = True

        model.config.tune_st_mlp_adapter = training_args.tune_st_mlp_adapter = model_args.tune_st_mlp_adapter
        if model_args.tune_st_mlp_adapter:
            print('model_args.tune_st_mlp_adapter==True')
            model.requires_grad_(False)
        for p in model.get_model().st_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().st_projector_sh.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_1.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_2.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_3.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_4.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_5.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_6.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_7.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_8.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_9.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_10.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_11.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_12.parameters():
            p.requires_grad = True
        for p in model.st_pred_linear_dispatch.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
        if not training_args.freeze_prompt_router:
            for p in model.prompt_router_sd.parameters():
                p.requires_grad = True
            for p in model.prompt_router_pems08.parameters():
                p.requires_grad = True
            for p in model.prompt_router_sz.parameters():
                p.requires_grad = True
            for p in model.prompt_router_urbanev.parameters():
                p.requires_grad = True
            for p in model.prompt_router_sh.parameters():
                p.requires_grad = True
            
        if training_args.bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
            bnb_model_from_pretrained_args.update(dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
                )
            ))
        if training_args.bits in [4, 8]:
            print('training_args.bits in [4, 8]')
            model.config.torch_dtype = (
                torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        model.config.freeze_st_mlp_adapter = training_args.freeze_st_mlp_adapter
        if training_args.freeze_st_mlp_adapter:
            print('model.config.freeze_st_mlp_adapter')
            for p in model.get_model().st_projector.parameters():
                p.requires_grad = False
            for p in model.get_model().st_projector_sh.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().st_projector.to(dtype=compute_dtype, device=training_args.device)
            model.get_model().st_projector_sh.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_1.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_2.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_3.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_4.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_5.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_6.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_7.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_8.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_9.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_10.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_11.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_12.to(dtype=compute_dtype, device=training_args.device)
            model.st_pred_linear_dispatch.to(dtype=compute_dtype, device=training_args.device)

        model.config.use_st_start_end = data_args.use_st_start_end = model_args.use_st_start_end
        training_args.use_st_start_end = model_args.use_st_start_end
        model.config.sep_st_conv_front = data_args.sep_st_conv_front
        print('use_st_start_end', training_args.use_st_start_end, 'sep_st_conv_front', model.config.sep_st_conv_front)
        model.initialize_st_tokenizer(use_st_start_end=model_args.use_st_start_end, tokenizer=tokenizer,
                                      device=training_args.device,
                                      tune_st_mlp_adapter=model_args.tune_st_mlp_adapter,
                                      pretrain_st_mlp_adapter=model_args.pretrain_st_mlp_adapter)
        
        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
        
        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(
                        len(params_no_grad), params_no_grad))
                else:
                    print(
                        '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                            len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print(
                    "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)

                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_stdata_module(tokenizer=tokenizer, data_args=data_args,batch_size=training_args.per_device_train_batch_size)

    trainer = STChatTrainer(model=model,
                            tokenizer=tokenizer,
                            args=training_args,
                            **data_module)
    
    print('************************** parameters: #', sum(p.numel() for p in model.parameters() if p.requires_grad))
    tuned_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            tuned_params.append(name)
    print(tuned_params)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model = model.merge_and_unload()
            model.save_pretrained(os.path.join(training_args.output_dir, 'full_model'))
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
