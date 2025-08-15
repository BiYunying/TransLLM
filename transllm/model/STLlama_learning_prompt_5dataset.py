#    Copyright 2023 Haotian Liu
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
import pandas as pd
from typing import List, Optional, Tuple, Union
from fastdtw import fastdtw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import os
from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM
import re
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transllm.model.st_layers.ST_Encoder import ST_Enc, parse_args
import json
import os.path as osp
import glob
import numpy as np 
from transllm.model.utils import preprocess_ST,preprocess
import copy

IGNORE_INDEX = -100
DEFAULT_STHIS_TOKEN = "<ST_EMB>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    mae_loss = torch.abs(true - pred)
    return torch.mean(mae_loss)

def scaler_mae_loss(scaler=None, mask_value=None):
    def loss(preds, labels, mask=None):
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def MSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = (true != mask_value)
        mse = ((pred - true) ** 2)[mask].mean()
    else:
        mse = ((pred - true) ** 2).mean()
    return mse

def scaler_mse_loss(scaler=None, mask_value=None):
    def loss(preds, labels, mask=None):
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if mask is not None:
            preds = preds * mask
            labels = labels * mask
        mse = MSE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mse
    return loss
class STLlamaConfig(LlamaConfig):
    model_type = "STLlama"

class STPretrainConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def load_model_pretrained(model_name, pretrain_model_path):
    # load conig json
    print("************************", pretrain_model_path)
    assert osp.exists(osp.join(pretrain_model_path, 'config.json')), 'config.json missing'
    with open(osp.join(pretrain_model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    args = STPretrainConfig(config_dict)
    model = model_name(args)
    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.pkl'))
    state_dict = torch.load(pkl_files[0])
    # print(state_dict.keys())
    if 'logit_scale' in state_dict.keys():
        state_dict.pop('logit_scale')
    print('loading ST pre train model')
    model.load_state_dict(state_dict)

    return model, args

class STLlamaModel(LlamaModel):
    config_class = STLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(STLlamaModel, self).__init__(config)
        self.st_start_id0 = []
        self.st_start_id1 = []
        self.st_start_id2 = []
        self.st_start_id3 = []
        self.st_end_id0 = []
        self.st_end_id1 = []
        self.st_end_id2 = []
        self.st_end_id3 = []
        self.pre_STE = None
        if hasattr(config, "st_tower"):
            if self.config.st_tower == "ST_Encoder":
                args = parse_args()
            
                self.st_tower = self.make_stmodel(args)
                filename = self.config.pretrain_ST_model_path
                state_dict = torch.load(filename, map_location="cpu")
                missing_keys, unexpected_keys = self.st_tower.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    print("Missing keys:", missing_keys)
                if unexpected_keys:
                    print("Unexpected keys:", unexpected_keys)
        if hasattr(config, "use_st_proj"):
            self.st_projector = nn.Linear(self.config.st_hidden_size*2, config.hidden_size)
            self.st_projector_sh = nn.Linear(self.config.st_hidden_size*2*12, self.config.hidden_size)
    
    def make_stmodel(self,args):
        model = ST_Enc(
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    device=args.device,
                    )
        return model
    
    def set_st_tower(self):
        st_tower = getattr(self, 'st_tower', None)
        if type(st_tower) is list:
            st_tower = st_tower[0]

        st_tower = st_tower.to(dtype=torch.float32)
        model_state_dict = st_tower.state_dict()
        loaded_state_dict = torch.load(self.config.pretrain_ST_model_path)
        for name, param in loaded_state_dict.items():
            model_state_dict[name].copy_(param)
        return st_tower

    def get_st_tower(self):
        st_tower = getattr(self, 'st_tower', None)
        if type(st_tower) is list:
            st_tower = st_tower[0]
        return st_tower

    def initialize_st_modules(self, st_tower, st_select_layer,
                                 pretrain_st_mlp_adapter=None, fsdp=None):  # TODO: modify this function
        self.config.st_tower = st_tower

        if hasattr(self, 'st_tower'):
            if self.config.st_tower == "ST_Encoder":
                args = parse_args()
                torch.backends.cudnn.enabled = False

                st_tower = self.make_stmodel(args)
                loaded_state_dict = torch.load(self.config.pretrain_ST_model_path)
                model_state_dict = st_tower.state_dict()
                for name, param in loaded_state_dict.items():
                    model_state_dict[name].copy_(param)
                st_tower.requires_grad_(False)
        else:
            st_tower = self.st_tower

        if fsdp is not None and len(fsdp) > 0:
            self.st_tower = [st_tower]
        else:
            self.st_tower = st_tower

        self.config.use_st_proj = True
        self.config.st_select_layer = st_select_layer

        if not hasattr(self, 'st_projector'):
            self.st_projector = nn.Linear(self.config.st_hidden_size*2, self.config.hidden_size)
            self.st_projector_sh = nn.Linear(self.config.st_hidden_size*2*12, self.config.hidden_size)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            st_data_x: Optional[list] = None,
            st_data_x_waiting: Optional[list] = None,
            st_data_y: Optional[list] = None,
            st_data_y_waiting: Optional[list] = None,
            region_start: Optional[int] = -1,
            region_end: Optional[int] = -1,
            return_dict: Optional[bool] = None,
            sp_matrix: Optional[torch.LongTensor] = None,
            se_matrix: Optional[torch.LongTensor] = None,
            se_matrix_waiting: Optional[torch.LongTensor] = None,
            nodes_feature: Optional[torch.LongTensor] = None,
            cur_token_len: Optional[torch.LongTensor] = None,
            data_type: Optional[int] = 0,
            neighbors: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if len(st_data_x) > 1:
            st_data_x = torch.cat(st_data_x, dim=0)
            st_data_y = torch.cat(st_data_y, dim=0)
            if st_data_x_waiting[0] != None:
                st_data_x_waiting = torch.cat(st_data_x_waiting, dim=0)
                st_data_y_waiting = torch.cat(st_data_y_waiting, dim=0)

        st_tower = self.get_st_tower()
        if st_tower is not None and (input_ids.shape[1] != 1 or self.training) and st_data_x is not None:
            if type(st_data_x) is list:
                pre_STE, STE_out = st_tower(st_data_x[0][..., :3],sp_matrix[0],se_matrix[0],data_type,nodes_feature[0])
                if st_data_x_waiting[0] is not None:
                    pre_STE_waiting, STE_out_waiting = st_tower(st_data_x_waiting[0][..., :3].cuda(),sp_matrix[0],se_matrix_waiting[0],4,nodes_feature[0])
                B = STE_out.size(0)
                if neighbors[0] is not None:
                    start = region_start[0]
                    end = region_end[0]
                    matched_rows = neighbors[0][neighbors[0][:, 0] == start]
                    matched_rows_tensor = torch.as_tensor(matched_rows[0], device=STE_out.device).long()
                    selected = STE_out[:, :, matched_rows_tensor, :]
                    region_select_out = selected.permute(0, 2, 1, 3).reshape(selected.size(0), selected.size(2), -1).to(torch.bfloat16)
                    selected_waiting = STE_out_waiting[:, :, matched_rows_tensor, :]
                    region_select_out_waiting = selected_waiting.permute(0, 2, 1, 3).reshape(selected_waiting.size(0), selected_waiting.size(2), -1).to(torch.bfloat16)
                    st_projector_out = self.st_projector_sh(region_select_out)
                    st_projector_out_waiting = self.st_projector_sh(region_select_out_waiting)
                else:
                    if STE_out.shape[2] >= 1:
                        region_select_out = STE_out[:, :, region_start[0]:region_end[0], :].squeeze(2).to(torch.bfloat16)
                        st_projector_out = self.st_projector(region_select_out)
            else:
                pre_STE, STE_out = st_tower(st_data_x[..., :3],sp_matrix[0],se_matrix[0],data_type,nodes_feature[0])
                if st_data_x_waiting[0] is not None:
                    pre_STE_waiting, STE_out_waiting = st_tower(st_data_x_waiting[..., :3],sp_matrix[0],se_matrix_waiting[0],4,nodes_feature[0])
                B = STE_out.size(0)
                region_select_out = []
                region_select_out_waiting = []
                for b in range(B):
                    start = region_start[b]
                    end = region_end[b]
                    # [T, start:end, F]
                    if neighbors[0] is not None:
                        matched_rows = neighbors[0][neighbors[0][:, 0] == start]
                        selected = STE_out[b, :, matched_rows[0].long(), :]
                        region_select_out.append(selected)
                        selected_waiting = STE_out_waiting[b, :, matched_rows[0].long(), :]
                        region_select_out_waiting.append(selected_waiting)
                    else:
                        selected = STE_out[b, :, start:end, :]
                        region_select_out.append(selected)

                if neighbors[0] is not None:
                    region_select_out = torch.stack(region_select_out, dim=0).permute(0, 2, 1, 3).reshape(B, region_select_out[0].shape[1], -1).to(torch.bfloat16)
                    st_projector_out = self.st_projector_sh(region_select_out)
                    region_select_out_waiting = torch.stack(region_select_out_waiting, dim=0).permute(0, 2, 1, 3).reshape(B, region_select_out_waiting[0].shape[1], -1).to(torch.bfloat16)
                    st_projector_out_waiting = self.st_projector_sh(region_select_out_waiting)
                else:
                    region_select_out = torch.stack(region_select_out, dim=0).squeeze(2).to(torch.bfloat16)
                    st_projector_out = self.st_projector(region_select_out)
            self.pre_STE = pre_STE

            new_input_embeds = []
            cur_st_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                cur_st_features = st_projector_out[cur_st_idx]
                cur_st_features = cur_st_features.reshape(cur_st_features.shape[0], -1)

                num_patches = cur_token_len[cur_st_idx]
                if (cur_input_ids == st_tower.config.st_start_token).sum() != (
                        cur_input_ids == st_tower.config.st_end_token).sum():
                    raise ValueError("The number of st start tokens and st end tokens should be the same.")
                st_start_tokens = torch.where(cur_input_ids == st_tower.config.st_start_token)[0]
                st_end_tokens = torch.where(cur_input_ids == st_tower.config.st_end_token)[0]

                if st_start_tokens.shape[0] == 3:
                    if neighbors[0] is not None:
                        st_start_token_pos1 = st_start_tokens[0]
                        st_start_token_pos2 = st_start_tokens[1]
                        st_start_token_pos3 = st_start_tokens[2]
                        self.st_start_id0.append(st_start_token_pos1)
                        self.st_start_id1.append(st_start_token_pos2)
                        self.st_start_id2.append(st_start_token_pos3)
                        self.st_end_id0.append(st_end_tokens[0])
                        self.st_end_id1.append(st_end_tokens[1])
                        self.st_end_id2.append(st_end_tokens[2])
                        cur_st_features_waiting = st_projector_out_waiting[cur_st_idx]
                        cur_st_features_waiting = cur_st_features_waiting.reshape(cur_st_features_waiting.shape[0], -1)
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos1 + 1],
                                                            cur_st_features[0:num_patches],
                                                            cur_input_embeds[st_start_token_pos1 + num_patches + 1:st_start_token_pos2],
                                                            cur_input_embeds[st_start_token_pos2:st_start_token_pos2 + 1],
                                                            cur_st_features_waiting[0:num_patches],
                                                            cur_input_embeds[st_start_token_pos2 + num_patches + 1:]),dim=0)
                        cur_st_idx += 1
                    else:
                        st_start_token_pos1 = st_start_tokens[0]
                        st_start_token_pos2 = st_start_tokens[1]
                        st_start_token_pos3 = st_start_tokens[2]
                        self.st_start_id0.append(st_start_token_pos1)
                        self.st_start_id1.append(st_start_token_pos3)
                        self.st_end_id0.append(st_end_tokens[0])
                        self.st_end_id1.append(st_end_tokens[2])
                        if cur_input_ids[
                            st_start_token_pos1 + num_patches + 1] != st_tower.config.st_end_token:
                            raise ValueError("The st end token should follow the st start token.")

                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos1].detach(),
                                                            cur_input_embeds[st_start_token_pos1:st_start_token_pos1 + 1],
                                                            cur_st_features[0:num_patches],
                                                            cur_input_embeds[st_start_token_pos1 + num_patches + 1:st_start_token_pos1 + num_patches + 2],
                                                            cur_input_embeds[st_start_token_pos1 + num_patches + 2:st_start_token_pos2].detach(),
                                                            cur_input_embeds[st_start_token_pos2:st_start_token_pos2 + num_patches + 2],
                                                            cur_input_embeds[st_start_token_pos2 + num_patches + 2:st_start_token_pos3].detach(),
                                                            cur_input_embeds[st_start_token_pos3:st_start_token_pos3 + num_patches + 2],
                                                            cur_input_embeds[st_start_token_pos3 + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos1 + 1],
                                                            cur_st_features[0:num_patches],
                                                            cur_input_embeds[st_start_token_pos1 + num_patches + 1:]), dim=0)
                        cur_st_idx += 1
                elif st_start_tokens.shape[0] == 4:
                    st_start_token_pos1 = st_start_tokens[0]
                    st_start_token_pos2 = st_start_tokens[1]
                    st_start_token_pos3 = st_start_tokens[2]
                    st_start_token_pos4 = st_start_tokens[3]
                    self.st_start_id0.append(st_start_token_pos1)
                    self.st_start_id1.append(st_start_token_pos2)
                    self.st_start_id2.append(st_start_token_pos4)
                    self.st_end_id0.append(st_end_tokens[0])
                    self.st_end_id1.append(st_end_tokens[1])
                    self.st_end_id2.append(st_end_tokens[3])
                    if cur_input_ids[
                        st_start_token_pos1 + num_patches + 1] != st_tower.config.st_end_token:
                        raise ValueError("The st end token should follow the st start token.")
                    cur_st_features_waiting = st_projector_out_waiting[cur_st_idx]
                    cur_st_features_waiting = cur_st_features_waiting.reshape(cur_st_features_waiting.shape[0], -1)
                    if orig_embeds_params is not None:
    
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos1].detach(),
                                                        cur_input_embeds[st_start_token_pos1:st_start_token_pos1 + 1],
                                                        cur_st_features[0:num_patches],
                                                        cur_input_embeds[st_start_token_pos1 + num_patches + 1:st_start_token_pos1 + num_patches + 2],
                                                        cur_input_embeds[st_start_token_pos1 + num_patches + 2:st_start_token_pos2].detach(),
                                                        cur_input_embeds[st_start_token_pos2:st_start_token_pos2 + 1],
                                                        cur_st_features_waiting[0:num_patches],
                                                        cur_input_embeds[st_start_token_pos2 + num_patches + 1:st_start_token_pos2 + num_patches + 2],
                                                        cur_input_embeds[st_start_token_pos2 + num_patches + 2:st_start_token_pos3].detach(),
                                                        cur_input_embeds[st_start_token_pos3:st_start_token_pos3 + num_patches + 2],
                                                        cur_input_embeds[st_start_token_pos3 + num_patches + 2:st_start_token_pos4].detach(),
                                                        cur_input_embeds[st_start_token_pos4:st_start_token_pos4 + num_patches + 2],
                                                        cur_input_embeds[st_start_token_pos4 + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos1 + 1],
                                                          cur_st_features[0:num_patches],
                                                          cur_input_embeds[st_start_token_pos1 + num_patches + 1:st_start_token_pos2],
                                                          cur_input_embeds[st_start_token_pos2:st_start_token_pos2 + 1],
                                                          cur_st_features_waiting[0:num_patches],
                                                          cur_input_embeds[st_start_token_pos2 + num_patches + 1:]),dim=0)
                    cur_st_idx += 1
                else:
                    st_start_token_pos = st_start_tokens[0]       
                    self.st_start_id0.append(st_start_token_pos)
                    self.st_end_id0.append(st_end_tokens[0])
    
                    num_patches = cur_token_len[cur_st_idx]
                    if cur_input_ids[st_start_token_pos + num_patches + 1] != st_tower.config.st_end_token:
                        raise ValueError("The st end token should follow the st start token.")

                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos].detach(),
                                                          cur_input_embeds[st_start_token_pos:st_start_token_pos + 1],
                                                          cur_st_features[0:num_patches],
                                                          cur_input_embeds[st_start_token_pos + num_patches + 1:st_start_token_pos + num_patches + 2],
                                                          cur_input_embeds[st_start_token_pos + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:st_start_token_pos + 1],
                                                          cur_st_features[0:num_patches],
                                                          cur_input_embeds[st_start_token_pos + num_patches + 1:]),dim=0)
                    cur_st_idx += 1
                new_input_embeds.append(cur_new_input_embeds)

            assert cur_st_idx == len(st_projector_out)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(STLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class MultiSlotActorCriticRouter(nn.Module):
    def __init__(self, embedding_dim: int, num_prompts: int, num_slots: int, lr: float = 1e-4, gamma: float = 0.99):
        super().__init__()
        self.num_prompts = num_prompts
        self.num_slots = num_slots
        self.gamma = gamma

       # Build separate Actor and Critic for each slot
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_prompts)
            ) for _ in range(num_slots)
        ])

        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_slots)
        ])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Each slot individually records log_prob / value / reward
        self.saved_log_probs = [[] for _ in range(num_slots)]
        self.saved_values = [[] for _ in range(num_slots)]
        self.rewards = [[] for _ in range(num_slots)]

    def forward(self, embedding):
        probs_per_slot = []
        for actor in self.actors:
            logits = actor(embedding)  # [B, num_prompts]
            probs = F.softmax(logits, dim=-1)
            probs_per_slot.append(probs)
        return probs_per_slot  # List of [B, num_prompts]

    def select_prompts(self, embedding):
        selected = []    
        embedding = embedding.detach()
        for slot in range(self.num_slots):
            actor = self.actors[slot]
            critic = self.critics[slot]
            logits = actor(embedding)
            logits = logits.to(torch.float32)   # [B, num_prompts]
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()  # [B]
            log_prob = dist.log_prob(action)  # [B]

            value = critic(embedding).squeeze(-1)  # [B]
            selected.append(action)

            self.saved_log_probs[slot].append(log_prob)
            self.saved_values[slot].append(value)
        return torch.stack(selected, dim=1)  # [B, num_slots]

    def store_reward(self, rewards_per_slot: list):
        """
        rewards_per_slot: List[Tensor[B]]
        """
        for slot in range(self.num_slots):
            self.rewards[slot].append(rewards_per_slot[slot])

    def finish_episode(self):
        if self.saved_log_probs[0] == []:
            return
        total_loss = None
        for slot in range(self.num_slots):
            if len(self.rewards[slot]) == 0:
                continue

            rewards = torch.stack(self.rewards[slot])
            log_probs = torch.stack(self.saved_log_probs[slot])
            values = torch.stack(self.saved_values[slot])

            T, B = rewards.shape
            returns = torch.zeros_like(rewards)
            R = torch.zeros(B, device=rewards.device)
            for t in reversed(range(T)):
                R = rewards[t] + self.gamma * R
                returns[t] = R

            advantage = returns - values
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss + critic_loss

            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.saved_log_probs = [[] for _ in range(self.num_slots)]
        self.saved_values = [[] for _ in range(self.num_slots)]
        self.rewards = [[] for _ in range(self.num_slots)]



class STLlamaForCausalLM(LlamaForCausalLM):
    config_class = STLlamaConfig

    def __init__(self, config,num_prompts,num_slots):
        # super(LlamaForCausalLM, self).__init__(config)
        super().__init__(config)
        self.model = STLlamaModel(config)
        self.prompt_router_sd = MultiSlotActorCriticRouter(embedding_dim = 128*12,num_prompts=num_prompts,
                                          num_slots=num_slots)
        self.prompt_router_sz = MultiSlotActorCriticRouter(embedding_dim = 128*12,num_prompts=num_prompts,
                                          num_slots=num_slots)
        self.prompt_router_sh = MultiSlotActorCriticRouter(embedding_dim = 128*12,num_prompts=num_prompts,
                                          num_slots=num_slots)
        self.prompt_router_pems08 = MultiSlotActorCriticRouter(embedding_dim = 128*12,num_prompts=num_prompts,
                                          num_slots=num_slots)
        self.prompt_router_urbanev = MultiSlotActorCriticRouter(embedding_dim = 128*12,num_prompts=num_prompts,
                                          num_slots=num_slots)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.st_pred_linear_1 = nn.Linear(self.config.hidden_size, self.config.lin_hidden_size)#(4096,128)
        self.st_pred_linear_2 = nn.Linear(self.config.lin_hidden_size*2, 1)#(256,12)
        self.st_pred_linear_3 = nn.Linear(self.config.hidden_size, self.config.lin_hidden_size)

        self.st_pred_linear_4 = nn.Linear(self.config.hidden_size, self.config.lin_hidden_size)#(4096,128)
        self.st_pred_linear_5 = nn.Linear(self.config.lin_hidden_size*2, 1)#(256,12)
        self.st_pred_linear_6 = nn.Linear(self.config.hidden_size, self.config.lin_hidden_size)

        self.st_pred_linear_7 = nn.Linear(self.config.hidden_size, self.config.lin_hidden_size)#(4096,128)
        self.st_pred_linear_8 = nn.Linear(self.config.lin_hidden_size*2, 1)#(256,12)
        self.st_pred_linear_9 = nn.Linear(self.config.hidden_size, self.config.lin_hidden_size)

        self.st_pred_linear_10 = nn.Linear(self.config.hidden_size, self.config.lin_hidden_size)#(4096,128)
        self.st_pred_linear_11 = nn.Linear(self.config.lin_hidden_size*2, 1)#(256,12)
        self.st_pred_linear_12 = nn.Linear(self.config.hidden_size, self.config.lin_hidden_size)
        self.st_pred_linear_dispatch = nn.Linear(self.config.hidden_size,1)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.tokenizer = None
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.st_pre_res = []
        self.batch_num = 0
        self.current_dataset=None
        
        # Initialize weights and apply final processing
        self.post_init()
    def set_tokenizer(self,tokenizer):
        self.tokenizer = tokenizer
    def get_prompt_routing(self):
        return self.prompt_router.get_routing()
    
    def get_model(self):
        return self.model

    def get_st_tower(self):
        return self.get_model().get_st_tower()

    def set_st_tower(self):
        self.get_model().set_st_tower()

    def get_vision_tower(self):
        model = self.get_model()
        st_tower = model.st_tower
        if type(st_tower) is list:
            st_tower = st_tower[0]
        return st_tower

    def get_st_pre_res(self):
        return self.st_pre_res

    def reset_st_pre_res(self):
        self.st_pre_res = []
    
    @classmethod
    def extract_info(self,original_sentence: str):
        traffic_match = re.search(r"traffic flow values are \[([^\]]+)\]", original_sentence)
        traffic_values = traffic_match.group(1).strip() if traffic_match else "unknown"

        # 提取历史时间，不包含 'with ...'
        history_time_match = re.search(
            r"The recording time of the historical data is '([^']*?)(?= with data points recorded)", original_sentence)
        history_time = history_time_match.group(1).strip().rstrip(',') if history_time_match else "unknown"

        # 提取未来时间段
        future_time_match = re.search(
            r"the next \d+ time steps during the time period of '([^']*?)(?= with data points recorded)", original_sentence)
        future_time = future_time_match.group(1).strip().rstrip(',') if future_time_match else "unknown"

        return traffic_values, history_time, future_time
    def extract_info_sh(self,sentence: str):
        empty_match = re.search(r"Current empty taxi count per grid: \[(.*?)\]", sentence)
        if empty_match:
            empty_counts = list(map(int, empty_match.group(1).split()))
        else:
           raise ValueError("Empty vehicle count information not found")

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
   
   
    def replace_prompt_sd(self,routing_info: torch.Tensor, original_sentence: str,st_data_xh_tmp,st_data_xd_tmp,st_data_xw_tmp) -> str:
        traffic_values, history_time, future_time = self.extract_info(original_sentence=original_sentence)
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

    def replace_prompt_sz(self, routing_info: torch.Tensor, original_sentence: str,
                      st_data_xh_tmp, st_data_xd_tmp, st_data_xw_tmp) -> str:
        traffic_values, history_time, future_time = self.extract_info(original_sentence=original_sentence)
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
    
    def replace_prompt_sh(self, routing_info: torch.Tensor, original_sentence: str, st_data_xh_tmp) -> str:
        empty_counts, current_time, dispatch_time = self.extract_info_sh(sentence=original_sentence)

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
    def simulate_dispatch_reward(self, empty_counts, idle_counts, demand_counts, probs, real_prob, eps=1e-6):
        """
        empty_counts: Tensor[B, 9]  Current number of empty vehicles in each grid (only Grid 0 is dispatchable)
        demand_counts: Tensor[B, 9] Current demand in each grid
        probs: Tensor[B, 9]  Dispatch probabilities (dispatching from Grid 0 to 9 grids, including itself)
        """

        # 1. Dispatch empty vehicles from Grid 0
        dispatched = empty_counts[:, 0:1].float() * probs.squeeze(2).squeeze(1)  # [B,9]

        # 2. Total empty vehicles in each grid = local + dispatched
        total_empty = idle_counts + dispatched

         # 3. Reward per grid: matching success rate (current demand * proportion of dispatched empty vehicles) - empty trip cost
        matched = demand_counts * (dispatched / (total_empty + eps))  # [B,9]

        # Empty trip distance penalty: calculated independently for each grid
        penalty_weight = torch.tensor(
            [0.5, 1.5, 1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 1.5],
            device=empty_counts.device
        ).view(1, 9)  # [1,9]
        penalty = (dispatched * penalty_weight) / (empty_counts[:, 0:1] + eps)  # [B,9]

        # Reward per grid: matching score - empty trip penalty
        per_grid_reward = 2.0*matched / (empty_counts[:, 0:1] + eps)- 0.05 * penalty# [B,9]

        probs = probs.squeeze(1).squeeze(1)  # [B,9]
        real_prob = real_prob.squeeze(1).squeeze(1)  # [B,9]

        probs = probs / (probs.sum(dim=1, keepdim=True) + eps)
        real_prob = real_prob / (real_prob.sum(dim=1, keepdim=True) + eps)

        cdf_p = torch.cumsum(probs, dim=1)
        cdf_q = torch.cumsum(real_prob, dim=1)
        wasserstein_distance = torch.abs(cdf_p - cdf_q).sum(dim=1)  # [B]
        return per_grid_reward, wasserstein_distance  # per_grid_reward shape: [B,9]





    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            sources: Optional[list] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            st_data_x: Optional[list] = None,
            st_data_xd: Optional[list] = None,
            st_data_xw: Optional[list] = None,
            st_data_y: Optional[list] = None,
            mean: Optional[list] = None,
            std: Optional[list] = None,
            st_data_x_waiting: Optional[list] = None,
            st_data_y_waiting: Optional[list] = None,
            mean_waiting: Optional[list] = None,
            std_waiting: Optional[list] = None,
            region_start: Optional[int] = -1,
            region_end: Optional[int] = -1,
            return_dict: Optional[bool] = None,
            sp_matrix: Optional[torch.LongTensor] = None,
            se_matrix: Optional[torch.LongTensor] = None,
            se_matrix_waiting: Optional[torch.LongTensor] = None,
            nodes_feature: Optional[torch.LongTensor] = None,
            patchlist: Optional[list] = None,
            neighbors: Optional[torch.Tensor] = None,
            real_prob: Optional[list] = None,
            empty_counts: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = next(self.parameters()).device
        data_type =0
        if sources != None:
            if sources[0][0]["id"].split('_')[1]=='SD':
                dataset='SD'
                data_type=1
            elif sources[0][0]["id"].split('_')[1]=='SZ':
                dataset='SZ'
                data_type =2
            elif sources[0][0]["id"].split('_')[1]=='SH':
                dataset='SH'
                data_type =3
            elif sources[0][0]["id"].split('_')[1]=='urbanev':
                dataset='urbanev'
                data_type =5
            elif sources[0][0]["id"].split('_')[1]=='pems08':
                dataset='pems08'
                data_type =6
        if input_ids == None:
            input_ids_batch = []
            labels_batch = []
            self.batch_num+=1
            if len(st_data_x) > 1:
                st_data_x_copy = copy.deepcopy(st_data_x)
                st_data_x_copy = torch.cat(st_data_x_copy, dim=0)
                mean=mean[0]
                std=std[0]
                st_data_x1_tmp = st_data_x_copy * std + mean
                if dataset=='SH':
                    st_data_xd_copy = None
                    st_data_xw_copy = None
                    real_prob = torch.cat(real_prob, dim=0)
                else:
                    st_data_xd_copy = copy.deepcopy(st_data_xd)
                    st_data_xw_copy = copy.deepcopy(st_data_xw)
                    st_data_xd_copy = torch.cat(st_data_xd_copy, dim=0)
                    st_data_xw_copy = torch.cat(st_data_xw_copy, dim=0)
                    st_data_xd_copy = st_data_xd_copy * std + mean   
                    st_data_xw_copy = st_data_xw_copy * std + mean 

            else:            
                st_data_x_copy = copy.deepcopy(st_data_x)
            _, node_embedding = self.model.st_tower(st_data_x_copy[..., :3],sp_matrix[0],se_matrix[0],data_type,nodes_feature[0])
            region_select_out = []
            st_data_xh_tmp = []
            st_data_xd_tmp = []
            st_data_xw_tmp = []
            for b in range(node_embedding.shape[0]):
                start = region_start[b]
                end = region_end[b]

                selected = node_embedding[b, :, start:end, :]
                region_select_out.append(selected)
                st_data_xh_tmp.append(st_data_x1_tmp[b, :, start:end, :])
                if dataset != 'SH':
                    st_data_xd_tmp.append(st_data_xd_copy[b, :, start:end, :])
                    st_data_xw_tmp.append(st_data_xw_copy[b, :, start:end, :])

            region_select_out = torch.stack(region_select_out, dim=0).reshape(node_embedding.shape[0],-1).to(torch.bfloat16)
            patchlist=[]
            empty_counts = []
            if dataset =='SD':
                routing_info_sd = self.prompt_router_sd.select_prompts(region_select_out)
                self.current_dataset = 'SD'
            elif dataset =='SZ':
                routing_info_sz = self.prompt_router_sz.select_prompts(region_select_out)
                self.current_dataset = 'SZ'
            elif dataset =='SH':
                routing_info_sh = self.prompt_router_sh.select_prompts(region_select_out)
                self.current_dataset = 'SH'
            elif dataset =='pems08':
                routing_info_pems08 = self.prompt_router_pems08.select_prompts(region_select_out)
                self.current_dataset = 'pems08'
            elif dataset =='urbanev':
                routing_info_urbanev = self.prompt_router_urbanev.select_prompts(region_select_out)
                self.current_dataset = 'urbanev'
            for i in range(len(sources)):  
                if dataset=='SD':         
                    sources[i][0]["conversations"][0]['value'] = self.replace_prompt_sd(
                        routing_info_sd[i],
                        sources[i][0]["conversations"][0]['value'],
                        st_data_xh_tmp[i],st_data_xd_tmp[i],st_data_xw_tmp[i]
                    )
                    cur_token_len = 12
                    output_token_len = 12

                elif dataset=='SZ':
                    sources[i][0]["conversations"][0]['value'] = self.replace_prompt_sz(
                        routing_info_sz[i],
                        sources[i][0]["conversations"][0]['value'],
                        st_data_xh_tmp[i],st_data_xd_tmp[i],st_data_xw_tmp[i]
                    )
                    cur_token_len = 12
                    output_token_len = 12
                elif dataset=='SH':
                    sources[i][0]["conversations"][0]['value'],empty_count = self.replace_prompt_sh(
                        routing_info_sh[i],
                        sources[i][0]["conversations"][0]['value'],
                        st_data_xh_tmp[i]
                    )
                    cur_token_len = 9
                    output_token_len = 9
                    empty_counts.append(torch.tensor(empty_count,device=device))
                elif dataset=='pems08':
                    sources[i][0]["conversations"][0]['value'] = self.replace_prompt_sd(
                        routing_info_pems08[i],
                        sources[i][0]["conversations"][0]['value'],
                        st_data_xh_tmp[i],st_data_xd_tmp[i],st_data_xw_tmp[i]
                    )
                    cur_token_len = 12
                    output_token_len = 12
                elif dataset=='urbanev':
                    sources[i][0]["conversations"][0]['value'] = self.replace_prompt_sz(
                        routing_info_urbanev[i],
                        sources[i][0]["conversations"][0]['value'],
                        st_data_xh_tmp[i],st_data_xd_tmp[i],st_data_xw_tmp[i]
                    )
                    cur_token_len = 12
                    output_token_len = 12

                st_cfg = {'use_st_start_end': True}

                patchlist.append(cur_token_len)
                sources[i] = preprocess_ST(
                    copy.deepcopy([e["conversations"] for e in sources[i]]),
                    st_cfg, cur_token_len,output_token_len
                )

                data_dict = preprocess(sources[i], self.tokenizer)
                input_ids = data_dict["input_ids"][0]
                labels = data_dict["labels"][0]
                input_ids_batch.append(input_ids)
                labels_batch.append(labels)

            input_ids_batch = torch.nn.utils.rnn.pad_sequence(
                input_ids_batch, batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            ).to(device)

            labels_batch = torch.nn.utils.rnn.pad_sequence(
                labels_batch, batch_first=True,
                padding_value=IGNORE_INDEX
            ).to(device)

            attention_mask = input_ids_batch.ne(self.tokenizer.pad_token_id).to(device)
        else:
            input_ids_batch = input_ids
            labels_batch = labels
            patchlist = patchlist[0]

        outputs = self.model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            st_data_x=st_data_x,
            st_data_y=st_data_y,
            st_data_x_waiting=st_data_x_waiting,
            st_data_y_waiting=st_data_y_waiting,
            region_start=region_start,
            region_end=region_end,
            sp_matrix=sp_matrix,
            se_matrix=se_matrix,
            se_matrix_waiting=se_matrix_waiting,
            nodes_feature=nodes_feature,
            cur_token_len=patchlist,
            data_type = data_type,
            neighbors = neighbors
        )

        feature_nums = 1
        hidden_states = outputs[0]
        batch_size = hidden_states.shape[0]
        st_pre_embs1_list = []
        st_pre_embs2_list = []
        if labels_batch is not None:
            for i in range(batch_size):
                # st_pre_embs1: from start_id0+1 to end_id0
                start1 = self.model.st_start_id0[i] + 1
                end1 = self.model.st_end_id0[i]
                emb1 = hidden_states[i, start1:end1, :].detach()
                st_pre_embs1_list.append(emb1)

                # st_pre_embs2: from start_id1+1 to start_id1 + feature_nums
                if dataset =='SH':
                    start2 = self.model.st_start_id2[i] + 1
                    end2 = self.model.st_end_id2[i]
                else:
                    start2 = self.model.st_start_id1[i] + 1
                    end2 = self.model.st_end_id1[i] 
                emb2 = hidden_states[i, start2:end2, :]
                st_pre_embs2_list.append(emb2)

            if dataset=='SD':
                st_pre_embs1 = torch.stack([x.reshape(-1, feature_nums, self.config.hidden_size) for x in st_pre_embs1_list], dim=0)
                st_pre_out1 = self.relu(self.st_pred_linear_1(st_pre_embs1))

                st_pre_embs2 = torch.stack([x.reshape(-1, feature_nums, self.config.hidden_size) for x in st_pre_embs2_list], dim=0)
                st_pre_out2 = self.relu(self.st_pred_linear_3(st_pre_embs2))

                st_pre_final = self.st_pred_linear_2(torch.cat([st_pre_out1, st_pre_out2], dim=-1)).reshape(st_pre_out1.shape[0],12,1,1)
            
            elif dataset =='SZ':
                st_pre_embs1 = torch.stack([x.reshape(-1, feature_nums, self.config.hidden_size) for x in st_pre_embs1_list], dim=0)
                st_pre_out1 = self.relu(self.st_pred_linear_4(st_pre_embs1))

                st_pre_embs2 = torch.stack([x.reshape(-1, feature_nums, self.config.hidden_size) for x in st_pre_embs2_list], dim=0)
                st_pre_out2 = self.relu(self.st_pred_linear_6(st_pre_embs2))

                st_pre_final = self.st_pred_linear_5(torch.cat([st_pre_out1, st_pre_out2], dim=-1)).reshape(st_pre_out1.shape[0],12,1,1)
            elif dataset =='SH':
                st_pre_embs2 = torch.stack([x.reshape(-1, feature_nums, self.config.hidden_size) for x in st_pre_embs2_list], dim=0)
               
                st_pre_final = self.st_pred_linear_dispatch(st_pre_embs2).permute(0,2,3,1)  # [B, 1, 1, 9]
                probs = torch.softmax(st_pre_final, dim=-1)

            elif dataset =='pems08':
                st_pre_embs1 = torch.stack([x.reshape(-1, feature_nums, self.config.hidden_size) for x in st_pre_embs1_list], dim=0)
                st_pre_out1 = self.relu(self.st_pred_linear_7(st_pre_embs1))

                st_pre_embs2 = torch.stack([x.reshape(-1, feature_nums, self.config.hidden_size) for x in st_pre_embs2_list], dim=0)
                st_pre_out2 = self.relu(self.st_pred_linear_9(st_pre_embs2))

                st_pre_final = self.st_pred_linear_8(torch.cat([st_pre_out1, st_pre_out2], dim=-1)).reshape(st_pre_out1.shape[0],12,1,1)
            elif dataset =='urbanev':
                st_pre_embs1 = torch.stack([x.reshape(-1, feature_nums, self.config.hidden_size) for x in st_pre_embs1_list], dim=0)
                st_pre_out1 = self.relu(self.st_pred_linear_10(st_pre_embs1))

                st_pre_embs2 = torch.stack([x.reshape(-1, feature_nums, self.config.hidden_size) for x in st_pre_embs2_list], dim=0)
                st_pre_out2 = self.relu(self.st_pred_linear_12(st_pre_embs2))

                st_pre_final = self.st_pred_linear_11(torch.cat([st_pre_out1, st_pre_out2], dim=-1)).reshape(st_pre_out1.shape[0],12,1,1)

            self.model.st_start_id0 = []
            self.model.st_start_id1 = []
            self.model.st_start_id2 = []
            self.model.st_end_id0 = []
            self.model.st_end_id1 = []
            self.model.st_end_id2 = []
        else:
            self.st_pre_res.append(hidden_states.clone())

        logits = self.lm_head(hidden_states)
        loss = None
        if labels_batch is not None:

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_batch[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            rec_loss = scaler_mae_loss(scaler=None, mask_value=None)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            label_stpre_list = []
            task_type_all_list = []
            shift_labels = shift_labels.to(shift_logits.device)
            if dataset =='SD' or dataset == 'SZ' or dataset == 'pems08' or dataset == 'urbanev':
                if len(st_data_y) > 1:
                    st_data_y = torch.cat(st_data_y, dim=0)

                    for i in range(batch_size):                    
                        label_stpre_list.append(st_data_y[i:i+1, :, region_start[i]:region_end[i], :feature_nums])
                        task_type_all_list.append(st_data_y[i:i+1, 0, region_start[i], -1])
                    
                    labels_stpre = torch.cat(label_stpre_list, dim=0).to(torch.bfloat16)
                    
                else:
                    labels_stpre = st_data_y[0][0:1, :, region_start[0]:region_end[0], :feature_nums].transpose(1, 2).to(torch.bfloat16)
                

                regress_idx_list = []

                regress_result_list = []

                for i in range(batch_size):        
                    regress_idx_list.append(i)
                        
                    regress_result_list.append(st_pre_final[i:i + 1, ...])
                regress_result = torch.cat(regress_result_list, dim=0)

                loss_regress = rec_loss(regress_result, labels_stpre)
                loss_per_sample = torch.mean(torch.abs(regress_result - labels_stpre), dim=(1, 2, 3))
                if dataset == 'SZ' or dataset == 'urbanev' :
                    loss_regress_scaling_factor = 1.0
                    loss_regress = loss_regress*loss_regress_scaling_factor
                loss = loss_fct(shift_logits, shift_labels) + loss_regress
            else:
                # ==== RL：dispatch reward ====
                real_prob_list=[]
                for i in range(batch_size):                    
                    real_prob_list.append(real_prob[i:i+1, :, region_start[i]:region_end[i], :])
                real_prob = torch.cat(real_prob_list, dim=0).to(torch.bfloat16)
                neighbor_mat = neighbors[0]  
                first_col = neighbor_mat[:, 0] 
                region_start_tensor = torch.tensor(region_start, device=neighbor_mat.device)
                mask = region_start_tensor.unsqueeze(1) == first_col.unsqueeze(0)  # [B, N]
                idx = mask.float().argmax(dim=1)  # [B]
                selected_rows_tensor = neighbor_mat[idx]  # [B, M]

                demand_index = selected_rows_tensor.long()
                if len(st_data_y) > 1:
                    st_data_y = torch.cat(st_data_y, dim=0) 
                    st_data_y_waiting = torch.cat(st_data_y_waiting, dim=0)                               
                else:
                    st_data_y = st_data_y[0]
                    st_data_y_waiting = st_data_y_waiting[0]
                demand_count=[]
                idle_count=[]
                for i in range(batch_size):                    
                    demand_count.append(st_data_y[i:i+1, :, demand_index[i], :1])
                    idle_count.append(st_data_y_waiting[i:i+1, :, demand_index[i], :1])
                demand_counts = torch.cat(demand_count, dim=0).squeeze(3).squeeze(1)
                idle_counts = torch.cat(idle_count, dim=0).squeeze(3).squeeze(1)
                
                empty_counts = torch.stack(empty_counts, dim=0)
                reward_per_grid, wasserstein_distance = self.simulate_dispatch_reward(
                empty_counts, idle_counts, demand_counts, probs, real_prob)  # reward_per_grid: [B,9]
                advantage = reward_per_grid
                reward_sum = reward_per_grid.sum(dim=1)
                entropy = -(probs * probs.log()).sum(dim=1).mean()
                loss1 = 0.5 -reward_sum.mean() - 0.008 * entropy
                loss_per_sample = -reward_per_grid.mean(dim=1)
                loss = loss_fct(shift_logits, shift_labels) + loss1*100 + wasserstein_distance.mean() *0.05
                print(f"Reward mean: {reward_per_grid.mean().item():.4f}, Reward std: {reward_per_grid.std().item():.4f}, Advantage std: {advantage.std().item():.4f}, Reinforce Loss: {loss1.item():.6f}, Wasserstein: {wasserstein_distance.mean().item():.6f}", flush=True)

            reward_slot = -loss_per_sample.detach() 

            rewards_per_slot = [reward_slot.clone() for _ in range(self.prompt_router_sd.num_slots)]

            if dataset == 'SD':
                self.prompt_router_sd.store_reward(rewards_per_slot)
            elif dataset == 'SZ':
                self.prompt_router_sz.store_reward(rewards_per_slot)
            elif dataset == 'SH':
                self.prompt_router_sh.store_reward(rewards_per_slot)
            elif dataset == 'pems08':
                self.prompt_router_pems08.store_reward(rewards_per_slot)
            elif dataset == 'urbanev':
                self.prompt_router_urbanev.store_reward(rewards_per_slot)

            print(loss_fct(shift_logits, shift_labels)) 

        if not return_dict:
            output = (logits,) + outputs[1:]
            print(loss.shape)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,

                "st_data_x": [kwargs.get("st_data_x", None)],
                "st_data_xd": [kwargs.get("st_data_xd", None)],
                "st_data_xw": [kwargs.get("st_data_xw", None)],
                "st_data_y": [kwargs.get("st_data_y", None)],
                "mean": [kwargs.get("mean", None)],
                "std": [kwargs.get("std", None)],
                "st_data_x_waiting": [kwargs.get("st_data_x_waiting", None)],
                "st_data_y_waiting": [kwargs.get("st_data_y_waiting", None)],
                "real_prob":  [kwargs.get("real_prob", None)],
                "region_start": [kwargs.get("region_start", None)],
                "region_end": [kwargs.get("region_end", None)],
                "nodes_feature": [kwargs.get("nodes_feature", None)],
                "empty_counts": [kwargs.get("empty_counts", None)],
                "sp_matrix": [kwargs.get("sp_matrix", None)],
                "se_matrix": [kwargs.get("se_matrix", None)],
                "se_matrix_waiting": [kwargs.get("se_matrix_waiting", None)],
                "patchlist": [kwargs.get("patchlist", None)],
                "neighbors": [kwargs.get("neighbors", None)]
            }
        )
        return model_inputs

    def reset_lm_head(self):
        self.get_input_embeddings().weight.data[-3:, :] = self.lm_head_add.weight.data

    def initialize_st_tokenizer(self, use_st_start_end, tokenizer, device,
                                   tune_st_mlp_adapter=False, pretrain_st_mlp_adapter=None):
        vision_config = self.get_st_tower().config
        vision_config.use_st_start_end = use_st_start_end
        tokenizer.add_tokens([DEFAULT_ST_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if use_st_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_ST_START_TOKEN, DEFAULT_ST_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.st_start_token, vision_config.st_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_ST_START_TOKEN, DEFAULT_ST_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_st_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

            if pretrain_st_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_st_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.st_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_ST_PATCH_TOKEN])[0]

AutoConfig.register("STLlama", STLlamaConfig)
AutoModelForCausalLM.register(STLlamaConfig, STLlamaForCausalLM)
