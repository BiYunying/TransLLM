import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn  as nn
import sys
import re
import copy
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)
from transllm.train.train_st_learning_prompt_5dataset import get_config, load_adj
from transllm.conversation import conv_templates, SeparatorStyle
from transllm.utils import disable_torch_init
from transllm.model import *
from transllm.model.utils import KeywordsStoppingCriteria
import json
from transllm.model.STLlama_learning_prompt_5dataset import STLlamaForCausalLM
import numpy as np
from tqdm import tqdm
import os.path as osp
import pickle
import pandas as pd
# import ray

DEFAULT_STHIS_TOKEN = "<ST_EMB>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"



def load_st(idx, instruct_item, st_data_all):

    sources = instruct_item

    region_start = int(sources["id"].split('_')[4])
    region_end = int(sources["id"].split('_')[5])
    i4data_all = int(sources["id"].split('_')[7])

    st_data_x = torch.Tensor(st_data_all[i4data_all]['data_x'])
    st_data_x_waiting = torch.Tensor(st_data_all[i4data_all]['data_x_waiting'])

    st_data_y = torch.Tensor(st_data_all[i4data_all]['data_y'])
    st_data_y_waiting = torch.Tensor(st_data_all[i4data_all]['data_y_waiting'])
    real_prob = torch.Tensor(st_data_all[i4data_all]['real_prob'])
    mean = torch.tensor(st_data_all[i4data_all]['mean'])
    std = torch.tensor(st_data_all[i4data_all]['std'])

    cur_token_len = 9
    
    return {
        'st_data_x': st_data_x,
        'st_data_y': st_data_y,
        'st_data_x_waiting': st_data_x_waiting,
        'st_data_y_waiting': st_data_y_waiting,
        'mean' : mean,
        'std' : std,
        'region_start': region_start,
        'region_end': region_end,
        'st_token_len': cur_token_len,
        'real_prob': real_prob
    }


def load_prompting_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data




def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    print('prompt_file_len', len(prompt_file))
    prompt_file = prompt_file[args.start_id:args.end_id]
    print('prompt_file_len', len(prompt_file))
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(split_list) == num_gpus:
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1:
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else:
        raise ValueError('error in the number of list')

    print('idx_list', idx_list)

    if osp.exists(args.output_res_path) is False:
        os.makedirs(args.output_res_path, exist_ok=True)

    for idx in range(len(idx_list) - 1):
        # print("1")
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]

        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        eval_model(
            args, prompt_file[start_idx:end_idx], start_split, end_split
            )

def extract_info_sh(sentence: str):
        empty_match = re.search(r"Current empty taxi count per grid: \[(.*?)\]", sentence)
        if empty_match:
            empty_counts = list(map(int, empty_match.group(1).split()))
        else:
            raise ValueError("Empty taxi count information not found")

        time_match = re.search(r"The current time is '(.*?)'", sentence)
        if time_match:
            current_time = time_match.group(1)
        else:
            raise ValueError("Current time information not found")

        dispatch_match = re.search(r"dispatch planning is for the period until '(.*?)'", sentence)
        if dispatch_match:
            dispatch_time = dispatch_match.group(1)
        else:
            raise ValueError("Dispatch time information not found")

        return empty_counts, current_time, dispatch_time
def replace_prompt_sh(routing_info: torch.Tensor, original_sentence: str, st_data_xh_tmp) -> str:
        empty_counts, current_time, dispatch_time = extract_info_sh(sentence=original_sentence)
        slot_0_intro = [
            "Your task is to redistribute empty taxis from the center grid (Grid 0) to nearby regions to maximize future passenger-taxi matching and successful ride completion rates, while minimizing unnecessary travel.",
            "You are tasked with relocating idle taxis from Grid 0 to surrounding areas to enhance ride matching success and minimize detours.",
            "Your responsibility is to ensure optimal dispatch from the central grid, maximizing ride completions while avoiding unnecessary travel.",
            "The goal is to efficiently dispatch idle taxis from the central region to surrounding grids, balancing supply-demand matching and reducing operational costs."
        ]

        slot_1_grid_structure = [
            "The grid is divided into a 3x3 region centered around the current grid (Grid 0). Each grid is indexed as follows: [0: center, 1: top-left, 2: top, 3: top-right, 4: left, 5: right, 6: bottom-left, 7: bottom, 8: bottom-right].",
            "The environment is structured as a 3-by-3 grid centered on Grid 0, with neighboring positions indexed from 1 to 8 as per directional orientation.",
            "A 3x3 matrix represents the spatial layout, with the agent at the center (Grid 0), and adjacent grids labeled according to compass directions.",
            "Grid 0 lies at the center of a 3×3 region; surrounding grids are indexed clockwise from top-left (1) to bottom-right (8)."
        ]

        slot_2_state_info = [
            f"Current empty taxi count per grid: {empty_counts}. The current time is '{current_time}', and the dispatch planning is for the period until '{dispatch_time}'.",
            f"At '{current_time}', empty taxis are distributed as {empty_counts}. Planning extends to '{dispatch_time}'.",
            f"Empty vehicle distribution: {empty_counts}; Forecast window: {current_time} to {dispatch_time}.",
            f"The current spatial supply is {empty_counts}. Planning covers the interval ending at '{dispatch_time}'."
        ]

        slot_3_reasoning_task = [
            "We use a pre-trained spatiotemporal encoder to represent the predicted demand and the number of idle vehicles in the 3*3 region during the dispatching period, denoted as <ST_EMB> and <ST_EMB>.  You must decide the probability distribution for dispatching taxis from Grid 0 to each destination (including staying in Grid 0). Consider demand-supply imbalance, proximity preferences, and integrate both current patterns and future predictions to optimize expected matching while minimizing travel costs. Output a 9-dimensional probability vector summing to 1.0 representing dispatch probabilities for [stay, top-left, top, top-right, left, right, bottom-left, bottom, bottom-right], and express it as <ST_PRE>.",
            "A pre-trained spatiotemporal model <ST_EMB> captures the future demand of the region, and <ST_EMB> denotes the corresponding idle vehicle distribution. Use it to guide the dispatch decision across 9 grid positions, balancing spatial proximity and supply-demand gaps. Output the probabilities as <ST_PRE>.",
            "Given the current status, predicted demand <ST_EMB> and predicted idle vehicle counts <ST_EMB>, compute a 9-way probability distribution for dispatch actions from Grid 0. Ensure the result is normalized and expressed as <ST_PRE>.",
            "Use <ST_EMB> and <ST_EMB> to understand upcoming demand and idle vehicles. Based on this, assign dispatch probabilities for Grid 0 to all 9 directions, and format them as <ST_PRE>."
        ]

        slots = [
            slot_0_intro,
            slot_1_grid_structure,
            slot_2_state_info,
            slot_3_reasoning_task
        ]

        prompt_parts = [slots[i][routing_info[i]] for i in range(4)]
        new_sentence = " ".join(prompt_parts)
        return new_sentence,empty_counts


def extract_info(original_sentence: str):
    traffic_match = re.search(r"traffic flow values are \[([^\]]+)\]", original_sentence)
    traffic_values = traffic_match.group(1).strip() if traffic_match else "unknown"

    history_time_match = re.search(
        r"The recording time of the historical data is '([^']*?)(?= with data points recorded)", original_sentence)
    history_time = history_time_match.group(1).strip().rstrip(',') if history_time_match else "unknown"

    future_time_match = re.search(
        r"the next \d+ time steps during the time period of '([^']*?)(?= with data points recorded)", original_sentence)
    future_time = future_time_match.group(1).strip().rstrip(',') if future_time_match else "unknown"

    return traffic_values, history_time, future_time

# @ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)

    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')

    model = STLlamaForCausalLM.from_pretrained(args.model_name, num_prompts=4, num_slots=4,torch_dtype=torch.bfloat16, use_cache=True,device_map=None,
                                                  low_cpu_mem_usage=False).to("cuda")

    model.set_st_tower()
    print('finish loading')
    args1 = get_config()
    args1.bs = 1
    nodes_feature1, sp_matrix1, se_matrix1, \
    nodes_feature2, sp_matrix2, se_matrix2, \
    sp_matrix3, se_matrix3, se_matrix4= load_adj(args1)
    
    use_st_start_end = getattr(model.config, "use_st_start_end", True)
    tokenizer.add_tokens([DEFAULT_ST_PATCH_TOKEN], special_tokens=True)
    if use_st_start_end:
        tokenizer.add_tokens([DEFAULT_ST_START_TOKEN, DEFAULT_ST_END_TOKEN], special_tokens=True)

    st_tower = model.get_model().st_tower


    st_config = st_tower.config
    st_config.st_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_ST_PATCH_TOKEN])[0]

    st_config.use_st_start_end = use_st_start_end
    if use_st_start_end:
        st_config.st_start_token, st_config.st_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_ST_START_TOKEN, DEFAULT_ST_END_TOKEN])


    res_data = []
    print(f'total: {len(prompt_file)}')
    with open(args.st_data_path, 'rb') as file:
        st_data_all = pickle.load(file)
    error_i = 0
    temp = 0
    output_file = osp.join(args.output_res_path, f'arxiv_test_res_{start_idx}_{end_idx}.json')
    with open(output_file, "w") as fout:
        fout.write("[\n")
    csv_file_path = "third_task_preprocess/square_grid_3km_shanghai.csv"
    df = pd.read_csv(csv_file_path)
    position_to_id = {}
    for _, row in df.iterrows():
        position_to_id[(row['row'], row['col'])] = row['grid_id']
    result = []
    valid_grids_data = [] 
    valid_count = 0

    # Define the relative positions of 9 directions (in order: center, top-left, top, top-right, left, right, bottom-left, bottom, bottom-right)
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

    neighbors = np.array(valid_grids_data) if valid_grids_data else np.array([]).reshape(0, 9)
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        st_dict = load_st(idx, instruct_item, st_data_all)
        st_token_len = st_dict['st_token_len']
        st_data_x = st_dict['st_data_x']
        st_data_y = st_dict['st_data_y']
        st_data_x_waiting = st_dict['st_data_x_waiting']
        st_data_y_waiting = st_dict['st_data_y_waiting']
        real_prob = st_dict['real_prob']
        region_start = st_dict['region_start']
        region_end = st_dict['region_end']
        std = st_dict['std']
        mean = st_dict['mean']

        st_data_xh_tmp = st_data_x * std + mean    
        
        st_data_xh_tmp = st_data_xh_tmp[0,:,region_start:region_end,:]


        qs = instruct_item["conversations"][0]["value"]
        original_sentence=instruct_item["conversations"][0]["value"]
        dataset = instruct_item["id"].split('_')[1]
        st_data_x_copy = copy.deepcopy(st_data_x).cuda()
        if dataset == "SH":
            node_feature = None
            sp_matrix = sp_matrix3.cuda()
            se_matrix = se_matrix3.cuda()
            _, node_embedding = model.model.st_tower(st_data_x_copy[..., :3],sp_matrix,se_matrix,3,node_feature)
            selected = node_embedding[:, :, region_start:region_end, :]
            routing_info = model.prompt_router_sh.select_prompts(selected.reshape(selected.shape[0],-1).to(torch.bfloat16)).squeeze(0)
            qs,empty_counts = replace_prompt_sh(routing_info,original_sentence,st_data_xh_tmp)
        patchlist = []
        cur_token_len = 9
        patchlist.append(cur_token_len)
        pre_token_len = 9
        replace_token = DEFAULT_ST_PATCH_TOKEN * cur_token_len
        replace_token = DEFAULT_ST_START_TOKEN + replace_token + DEFAULT_ST_END_TOKEN
        replace_token1 = DEFAULT_ST_PATCH_TOKEN * pre_token_len
        replace_token1 = DEFAULT_ST_START_TOKEN + replace_token1 + DEFAULT_ST_END_TOKEN
        qs = qs.replace(DEFAULT_STHIS_TOKEN, replace_token)
        qs = qs.replace(DEFAULT_STPRE_TOKEN, replace_token1)

        # if "v1" in args.model_name.lower():
        #     conv_mode = "stchat_v1"
        # else:
        #     raise ValueError('Don\'t support this model')
        conv_mode = "stchat_llama"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        formatted_conversation = conv.system
        formatted_conversation += f"<｜User｜>{qs}<｜Assistant｜>"
        conv.messages.append(formatted_conversation)
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        prompt = formatted_conversation
        # print(prompt)
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        attention_mask = (input_ids != 128256).long()


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                st_data_x=st_data_x.cuda(),
                st_data_y=st_data_y.cuda(),
                region_start=region_start,
                region_end=region_end,
                do_sample=True,
                attention_mask = attention_mask,
                pad_token_id=128256,
                temperature=0.01,
                max_new_tokens=256,
                nodes_feature = node_feature,
                sp_matrix = sp_matrix,
                se_matrix = se_matrix,
                se_matrix_waiting = se_matrix4.cuda(),
                patchlist=patchlist,
                mean = [mean],
                std = [std],
                st_data_x_waiting = st_data_x_waiting,
                st_data_y_waiting = st_data_x_waiting,
                real_prob = real_prob,
                empty_counts = empty_counts,
                neighbors = neighbors,
                stopping_criteria=[stopping_criteria])

            # Find the special tokens
            start_inx = torch.where(output_ids[0, :] == 128258)[0]
            end_inx = torch.where(output_ids[0, :] == 128259)[0]
            # Get hidden_states
            hidden_states = model.get_st_pre_res()
            hidden_states = torch.cat(hidden_states, dim=1)
            model.reset_st_pre_res()

            # Decode the token into the result
            batch_size = hidden_states.shape[0]

            feature_nums = 1
            if start_inx.shape[0] == 4 and end_inx.shape[0] == 4 and end_inx[3]-start_inx[3]==10:
                st_pre_embs2 = hidden_states[:,
                            start_inx[3]+1:end_inx[3],
                            :].reshape(batch_size, -1, feature_nums, model.config.hidden_size)
                probs = model.st_pred_linear_dispatch(st_pre_embs2).permute(0,2,3,1)  # [B, 1, 1, 9]
                probs = torch.softmax(probs, dim=-1).squeeze().tolist()
                real_prob = real_prob[:, :, region_start:region_end, :].squeeze().tolist()
                
                neighbor_mat = neighbors  
                first_col = neighbor_mat[:, 0] 
                region_start_tensor = torch.tensor(region_start)
                mask = region_start_tensor == first_col  # [B, N]
                idx = mask.float().argmax()  # [B]
                selected_rows_tensor = neighbor_mat[idx]  # [B, M]

                demand_index = selected_rows_tensor.astype(int)
                if len(st_data_y) > 1:
                    st_data_y = torch.cat(st_data_y, dim=0) 
                    st_data_y_waiting = torch.cat(st_data_y_waiting, dim=0)                               
                else:
                    st_data_y = st_data_y[0]
                    st_data_y_waiting = st_data_y_waiting[0]
                demand_count=[]
                idle_count=[]
                                 
                demand_count.append(st_data_y[ :, (demand_index[:]), :1])
                idle_count.append(st_data_y_waiting[ :, (demand_index[:]), :1])
                demand = torch.cat(demand_count, dim=0).squeeze(2).squeeze(0).tolist()
                waiting = torch.cat(idle_count, dim=0).squeeze(2).squeeze(0).tolist()
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
                res_data.append(
                        {"id": instruct_item["id"], "empty":empty_counts, "waiting": waiting, "demand": demand,
                            "probs": probs,"real_prob":real_prob}.copy())
                with open(osp.join(args.output_res_path, 'arxiv_test_res_{}_{}.json'.format(start_idx, end_idx)), "w") as fout:
                    json.dump(res_data, fout, indent=4)
            else:
                print('========error========')
                error_i = error_i + 1
                print(error_i)
    return


if __name__ == "__main__":
    output_model='./checkpoints/yourmodel'
    datapath='./data/prompt_data/SH_2015_test.json'
    st_data_path='./data/prompt_data/SH_2015_test_pkl.pkl'
    res_path='./result_test1/SH'
    start_id=0
    end_id=10824
    num_gpus=1

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=output_model)
    parser.add_argument("--prompting_file", type=str, default=datapath)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--st_data_path", type=str, default=st_data_path)
    parser.add_argument("--use_st_start_end", type=bool, default=True)
    parser.add_argument("--output_res_path", type=str, default=res_path)
    parser.add_argument("--num_gpus", type=int, default=num_gpus)

    parser.add_argument("--start_id", type=int, default=start_id)
    parser.add_argument("--end_id", type=int, default=end_id)

    args = parser.parse_args()

    run_eval(args, args.num_gpus)
