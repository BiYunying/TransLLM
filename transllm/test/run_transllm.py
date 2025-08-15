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
import json
import os.path as osp
import pickle

# import ray

DEFAULT_STHIS_TOKEN = "<ST_EMB>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"



def load_st(idx, instruct_item, st_data_all):

    sources = instruct_item

    region_start = int(sources["id"].split('_')[3])
    region_end = int(sources["id"].split('_')[4])
    i4data_all = int(sources["id"].split('_')[6])

    st_data_x = torch.Tensor(st_data_all[i4data_all]['data_x'])
    st_data_xd = torch.Tensor(st_data_all[i4data_all]['data_x_1d'])
    st_data_xw = torch.Tensor(st_data_all[i4data_all]['data_x_1w'])
    st_data_y = torch.Tensor(st_data_all[i4data_all]['data_y'])

    mean = torch.tensor(st_data_all[i4data_all]['mean'])
    std = torch.tensor(st_data_all[i4data_all]['std'])

    cur_token_len = 1

    return {
        'st_data_x': st_data_x,
        'st_data_xd': st_data_xd,
        'st_data_xw': st_data_xw,
        'st_data_y': st_data_y,
        'mean' : mean,
        'std' : std,
        'region_start': region_start,
        'region_end': region_end,
        'st_token_len': cur_token_len
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
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]

        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        eval_model(
            args, prompt_file[start_idx:end_idx], start_split, end_split
            )

def replace_prompt_sd(routing_info: torch.Tensor, original_sentence: str,st_data_xh_tmp,st_data_xd_tmp,st_data_xw_tmp) -> str:
        traffic_values, history_time, future_time = extract_info(original_sentence=original_sentence)
        st_data_xh = st_data_xh_tmp[:,0,0].int()
        st_data_xd = st_data_xd_tmp[:,0,0].int()
        st_data_xw = st_data_xw_tmp[:,0,0].int()
        slot_0_history_flow = [
            f"Using traffic flow recorded one hour ago {st_data_xh.tolist()}, we examine recent short-term variations to identify immediate trends.",
            
            f"To uncover daily temporal patterns, we utilize both the traffic from one hour ago {st_data_xh.tolist()} and from the same time yesterday {st_data_xd.tolist()}, enabling comparison across consecutive days.",
            
            f"Recognizing weekly rhythms, we analyze flow data from one hour ago {st_data_xh.tolist()} alongside that from the same time last week {st_data_xw.tolist()}, aiming to detect recurring weekly behaviors.",
            
            f"By jointly considering traffic observations from one hour ago {st_data_xh.tolist()}, the same time yesterday {st_data_xd.tolist()}, and the same time last week {st_data_xw.tolist()}, we explore both short-term fluctuations and long-term temporal dependencies."
        ]

        slot_1_time_context = [
            f"Based on these inputs, the task is to forecast traffic for the next hour starting from {future_time}, with each step representing a 5-minute interval to capture short-term shifts.",

            f"Grounded on these historical cues, we now look ahead from {future_time}, forecasting 12 future values at 5-minute intervals to reveal potential periodic trends.",

            f"Given the temporal scope of the input, we proceed to estimate future traffic patterns starting at {future_time}, carefully adjusting for time-of-day effects such as rush hours or quiet periods.",

            f"With this historical context in mind, the model will generate a 12-step forecast beginning at {future_time}, incorporating 5-minute resolution to synthesize both rapid changes and gradual trends."
        ]


        slot_2_task_description = [
            "To begin the forecasting process, we utilize a pretrained spatio-temporal encoder to capture dependencies within the 12-step prediction horizon: <ST_EMB>.",
            
            "In order to better understand the early dynamics of traffic flow, we incorporate 12 spatio-temporal embeddings that correspond to future time intervals: <ST_EMB>.",
            
            "Specifically, a sequence of 12 step tokens is employed to represent the spatio-temporal context within the 12-step prediction window: <ST_EMB>.",
            
            "To provide comprehensive temporal context, the spatio-temporal encoder embeds all 12 predicted intervals with dedicated representations, focusing on early-stage forecasting: <ST_EMB>."
        ]

        slot_3_answer_format = [  
            "Please reason step-by-step through both temporal patterns and spatial influences. After that, generate the predicted traffic volume for the next 12 time intervals using the token <ST_PRE>.",
            
            "First, analyze recent flow trends and any irregular fluctuations you observe. Then, estimate future traffic volumes and present your answer using <ST_PRE>.",
            
            "To guide your prediction, draw from common traffic behaviors—such as rush hour surges or off-peak periods—and conclude your response with the 12-step forecast in <ST_PRE>.",
            
            "Consider potential causes of congestion or unexpected volume shifts. Based on this reasoning, produce a stable and well-grounded 12-step forecast using <ST_PRE>."
        ]



        slots = [
            slot_0_history_flow,
            slot_1_time_context,
            slot_2_task_description,
            slot_3_answer_format
        ]
         
        prompt_parts = [slots[i][routing_info[i]] for i in range(4)]
        new_sentence = " ".join(prompt_parts)
        return new_sentence
def replace_prompt_sz(routing_info: torch.Tensor, original_sentence: str,
                      st_data_xh_tmp, st_data_xd_tmp, st_data_xw_tmp) -> str:
        traffic_values, history_time, future_time = extract_info(original_sentence=original_sentence)
        st_data_xh = st_data_xh_tmp[:, 0, 0].int()
        st_data_xd = st_data_xd_tmp[:, 0, 0].int()
        st_data_xw = st_data_xw_tmp[:, 0, 0].int()

        slot_0_history_flow = [
            f"Using charging load recorded one hour ago {st_data_xh.tolist()}, we investigate short-term dynamics to capture immediate fluctuations.",
            f"To explore daily charging behavior, we compare the load from one hour ago {st_data_xh.tolist()} and the same time yesterday {st_data_xd.tolist()}, helping detect day-over-day shifts.",
            f"To identify weekly demand patterns, we reference charging data from one hour ago {st_data_xh.tolist()} and the same time last week {st_data_xw.tolist()}, examining recurring temporal signals.",
            f"By jointly analyzing charging records from one hour ago {st_data_xh.tolist()}, yesterday {st_data_xd.tolist()}, and last week {st_data_xw.tolist()}, we model both near-term and long-range dependencies."
        ]

        slot_1_time_context = [
            f"Based on these temporal references, the task is to forecast charging load for the next hour starting from {future_time}, with 5-minute intervals capturing fine-grained changes.",
            f"Leveraging these time-aligned insights, we aim to predict the 12-step charging demand starting at {future_time}, revealing emerging patterns.",
            f"Considering the timing context, we now estimate near-future charging needs from {future_time} forward, incorporating common time-of-day usage trends.",
            f"With this multi-scale temporal context, the forecast begins at {future_time}, enabling the model to learn both abrupt shifts and smoother demand transitions."
        ]

        slot_2_task_description = [
            "To begin the prediction process, we utilize a pretrained spatio-temporal encoder that captures dependencies from the 12 future steps: <ST_EMB>.",
            "For better modeling of early-stage charging dynamics, 12patio-temporal embeddings are used to represent the future time intervals: <ST_EMB>.",
            "The encoder uses a sequence of 12 step tokens to represent spatio-temporal context over the prediction steps: <ST_EMB>.",
            "A complete spatio-temporal embedding of all 12 predicted intervals is used, focusing on the early part of the forecast: <ST_EMB>."
        ]
        
        slot_3_answer_format = [
            "Please reason step-by-step through both temporal rhythms and spatial correlations. Then generate the 12-step charging load forecast using <ST_PRE>.",
            "Begin by analyzing trends and irregularities in recent data. Based on your reasoning, generate your forecast using <ST_PRE>.",
            "Rely on knowledge of typical EV charging demand cycles—like peak periods—to guide your reasoning, then provide the result via <ST_PRE>.",
            "Consider both recurring patterns and unexpected shifts in demand. Then produce a robust 12-step forecast using <ST_PRE>."
        ]

        slots = [
            slot_0_history_flow,
            slot_1_time_context,
            slot_2_task_description,
            slot_3_answer_format
        ]

        prompt_parts = [slots[i][routing_info[i]] for i in range(4)]
        new_sentence = " ".join(prompt_parts)
        return new_sentence


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
    
    disable_torch_init()
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
    nodes_feature1,sp_matrix1,se_matrix1,\
            nodes_feature2,sp_matrix2,se_matrix2,\
                sp_matrix3,se_matrix3,se_matrix4,nodes_feature5,\
                    sp_matrix5,se_matrix5,sp_matrix6,se_matrix6=load_adj(args1)
   
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

    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        st_dict = load_st(idx, instruct_item, st_data_all)
        st_token_len = st_dict['st_token_len']
        st_data_x = st_dict['st_data_x']
        st_data_xd = st_dict['st_data_xd']
        st_data_xw = st_dict['st_data_xw']
        st_data_y = st_dict['st_data_y']
        region_start = st_dict['region_start']
        region_end = st_dict['region_end']
        std = st_dict['std']
        mean = st_dict['mean']

        st_data_xh_tmp = st_data_x * std + mean    
        st_data_xd_tmp = st_data_xd * std + mean   
        st_data_xw_tmp = st_data_xw * std + mean 
        
        st_data_xh_tmp = st_data_xh_tmp[0,:,region_start:region_end,:]
        st_data_xd_tmp = st_data_xd_tmp[0,:,region_start:region_end,:]
        st_data_xw_tmp = st_data_xw_tmp[0,:,region_start:region_end,:]

        qs = instruct_item["conversations"][0]["value"]
        original_sentence=instruct_item["conversations"][0]["value"]
        dataset = instruct_item["id"].split('_')[1]
        st_data_x_copy = copy.deepcopy(st_data_x).cuda()
        if dataset == "SD":
            node_feature = nodes_feature1.cuda()
            sp_matrix = sp_matrix1.cuda()
            se_matrix = se_matrix1.cuda()
            _, node_embedding = model.model.st_tower(st_data_x_copy[..., :3],sp_matrix,se_matrix,1,node_feature)
            selected = node_embedding[:, :, region_start:region_end, :]
            routing_info = model.prompt_router_sd.select_prompts(selected.reshape(selected.shape[0],-1).to(torch.bfloat16)).squeeze(0)
            qs = replace_prompt_sd(routing_info,original_sentence,st_data_xh_tmp,st_data_xd_tmp,st_data_xw_tmp)
        elif dataset == "SZ":
            node_feature = nodes_feature2.cuda()
            sp_matrix = sp_matrix2.cuda()
            se_matrix = se_matrix2.cuda()
            _, node_embedding = model.model.st_tower(st_data_x_copy[..., :3],sp_matrix,se_matrix,2,node_feature)
            selected = node_embedding[:, :, region_start:region_end, :]
            routing_info = model.prompt_router_sz.select_prompts(selected.reshape(selected.shape[0],-1).to(torch.bfloat16)).squeeze(0)
            qs = replace_prompt_sz(routing_info,original_sentence,st_data_xh_tmp,st_data_xd_tmp,st_data_xw_tmp)
        elif dataset == "pems08":
            node_feature = None
            sp_matrix = sp_matrix6.cuda()
            se_matrix = se_matrix6.cuda()
            _, node_embedding = model.model.st_tower(st_data_x_copy[..., :3],sp_matrix,se_matrix,6,node_feature)
            selected = node_embedding[:, :, region_start:region_end, :]
            routing_info = model.prompt_router_pems08.select_prompts(selected.reshape(selected.shape[0],-1).to(torch.bfloat16)).squeeze(0)
            qs = replace_prompt_sd(routing_info,original_sentence,st_data_xh_tmp,st_data_xd_tmp,st_data_xw_tmp)
        elif dataset == "urbanev":
            node_feature = nodes_feature5.cuda()
            sp_matrix = sp_matrix5.cuda()
            se_matrix = se_matrix5.cuda()
            _, node_embedding = model.model.st_tower(st_data_x_copy[..., :3],sp_matrix,se_matrix,5,node_feature)
            selected = node_embedding[:, :, region_start:region_end, :]
            routing_info = model.prompt_router_urbanev.select_prompts(selected.reshape(selected.shape[0],-1).to(torch.bfloat16)).squeeze(0)
            qs = replace_prompt_sz(routing_info,original_sentence,st_data_xh_tmp,st_data_xd_tmp,st_data_xw_tmp)

        patchlist = []
        cur_token_len = 12
        patchlist.append(cur_token_len)
        pre_token_len = 12
        replace_token = DEFAULT_ST_PATCH_TOKEN * cur_token_len
        replace_token = DEFAULT_ST_START_TOKEN + replace_token + DEFAULT_ST_END_TOKEN
        replace_token1 = DEFAULT_ST_PATCH_TOKEN * pre_token_len
        replace_token1 = DEFAULT_ST_START_TOKEN + replace_token1 + DEFAULT_ST_END_TOKEN
        qs = qs.replace(DEFAULT_STHIS_TOKEN, replace_token)
        qs = qs.replace(DEFAULT_STPRE_TOKEN, replace_token1)

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

        prompt = formatted_conversation

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
                patchlist=patchlist,
                mean = [mean],
                std = [std],
                st_data_xd = st_data_xd,
                st_data_xw = st_data_xw,
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
            if start_inx.shape[0] == 3 and end_inx.shape[0] == 3 and end_inx[2]-start_inx[2]==12:
                st_pre_embs1 = hidden_states[:,
                            start_inx[0]+1:end_inx[0],
                            :].detach().reshape(batch_size, -1, feature_nums, model.config.hidden_size)
                st_pre_out1 = model.relu(model.st_pred_linear_1(st_pre_embs1))


                st_pre_embs2 = hidden_states[:,
                            start_inx[2]+1:end_inx[2],
                            :].reshape(batch_size, -1, feature_nums, model.config.hidden_size)
                st_pre_out2 = model.relu(model.st_pred_linear_3(st_pre_embs2))

                st_pre_final = model.st_pred_linear_2(torch.cat([st_pre_out1, st_pre_out2], dim=-1)).reshape(st_pre_out1.shape[0],12,1,1)
                st_pre_infolow = st_pre_final[:, :, :, 0].squeeze().detach().cpu().tolist()


                x_in, y_in = st_data_x[:, :, region_start:region_end, 0].squeeze().tolist(), st_data_y[0, :, region_start:region_end,
                                                                                0].squeeze().tolist()
 
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
 
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
                res_data.append(
                        {"id": instruct_item["id"],  "x_in": x_in, "y_in": y_in,
                            "st_pre_infolow": st_pre_infolow}.copy())
                with open(osp.join(args.output_res_path, 'arxiv_test_res_{}_{}.json'.format(start_idx, end_idx)), "w") as fout:
                    json.dump(res_data, fout, indent=4)
            else:
                print('========error========')
                error_i = error_i + 1
                print(error_i)
    return


if __name__ == "__main__":

    output_model='./checkpoints/your_model'
    datapath='./data/prompt_data/pems08_test.json'
    st_data_path='./data/prompt_data/pems08_test_pkl.pkl'
    res_path='./result_test/pems08'
    start_id=0
    end_id=170*12
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
