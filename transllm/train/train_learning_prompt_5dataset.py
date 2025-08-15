# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
import transformers
import wandb

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)

from transllm.train.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

model_path = "./checkpoints/llama3-8b"
instruct_ds_sd = "./data/prompt_data/SD_2021_supervised.json"
st_data_path_sd = "./data/prompt_data/SD_2021_supervised_pkl.pkl"
instruct_ds_pems08 = "./data/prompt_data/pems08_supervised.json"
st_data_path_pems08 = "./data/prompt_data/pems08_supervised_pkl.pkl"
instruct_ds_sz = "./data/prompt_data/SZ_2022_supervised.json"
st_data_path_sz = "./data/prompt_data/SZ_2022_supervised_pkl.pkl"
instruct_ds_urbanev = "./data/prompt_data/urbanev_supervised.json"
st_data_path_urbanev = "./data/prompt_data/urbanev_supervised_pkl.pkl"
instruct_ds_sh = "./data/prompt_data/SH_2015_supervised.json"
st_data_path_sh = "./data/prompt_data/SH_2015_supervised_pkl.pkl"
pretra_ste = "ST_Encoder"
output_model = "./checkpoints/PEMS08"

wandb.init(mode="offline")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default=model_path, type=str)
parser.add_argument("--version", default="v2", type=str)
parser.add_argument("--data_path_sd", default=instruct_ds_sd, type=str)
parser.add_argument("--data_path_pems08", default=instruct_ds_pems08, type=str)
parser.add_argument("--data_path_sz", default=instruct_ds_sz, type=str)
parser.add_argument("--data_path_urbanev", default=instruct_ds_urbanev, type=str)
parser.add_argument("--data_path_sh", default=instruct_ds_sh, type=str)
parser.add_argument("--st_content", default="./TAXI.json", type=str)
parser.add_argument("--st_data_path_sd", default=st_data_path_sd, type=str)
parser.add_argument("--st_data_path_pems08", default=st_data_path_pems08, type=str)
parser.add_argument("--st_data_path_sz", default=st_data_path_sz, type=str)
parser.add_argument("--st_data_path_urbanev", default=st_data_path_urbanev, type=str)
parser.add_argument("--st_data_path_sh", default=st_data_path_sh, type=str)
parser.add_argument("--st_tower", default="ST_Encoder", type=str)
parser.add_argument("--tune_st_mlp_adapter", default=False, type=bool)
parser.add_argument("--st_select_layer", default=-2, type=int)
parser.add_argument("--use_st_start_end", action='store_true')
parser.add_argument("--bf16", default=True, type=bool)
parser.add_argument("--output_dir", default=output_model, type=str)
parser.add_argument("--num_train_epochs", default=1,type=int)
parser.add_argument("--per_device_train_batch_size", default=8, type=int)
parser.add_argument("--num_prompts", default=4, type=int)
parser.add_argument("--num_slots", default=4, type=int)
parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--evaluation_strategy", default="no", type=str)
parser.add_argument("--save_strategy", default="steps", type=str)
parser.add_argument("--save_steps", default=4800, type=int)
parser.add_argument("--save_total_limit", default=1, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--warmup_ratio", default=0.03, type=float)
parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
parser.add_argument("--logging_steps", default=1, type=int)
parser.add_argument("--tf32", default=True, type=bool)
parser.add_argument("--model_max_length", default=2048, type=int)
parser.add_argument("--gradient_checkpointing", default=True, type=bool)
parser.add_argument("--lazy_preprocess", default=True, type=bool)
parser.add_argument("--report_to", default="wandb", type=str)
parser.add_argument("--bits", default=8, type=int)
parser.add_argument("--lora_enable", default=True, type=bool)
parser.add_argument("--freeze_backbone", default=True, type=bool)
parser.add_argument("--freeze_prompt_router", default=False, type=bool)

args = parser.parse_args()

from transllm.train.train_st_learning_prompt_5dataset import ModelArguments, DataArguments, TrainingArguments

hf_parser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments)
)

model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses(
    args=[
        f"--model_name_or_path={args.model_name_or_path}",
        f"--version={args.version}",
        f"--data_path_sd={args.data_path_sd}",
        f"--data_path_pems08={args.data_path_pems08}",
        f"--data_path_sz={args.data_path_sz}",
        f"--data_path_urbanev={args.data_path_urbanev}",
        f"--data_path_sh={args.data_path_sh}",
        f"--st_content={args.st_content}",
        f"--st_data_path_sd={args.st_data_path_sd}",
        f"--st_data_path_pems08={args.st_data_path_pems08}",
        f"--st_data_path_sz={args.st_data_path_sz}",
        f"--st_data_path_urbanev={args.st_data_path_urbanev}",
        f"--st_data_path_sh={args.st_data_path_sh}",
        f"--st_tower={args.st_tower}",
        f"--tune_st_mlp_adapter={args.tune_st_mlp_adapter}",
        f"--st_select_layer={args.st_select_layer}",
        f"--use_st_start_end",
        f"--bf16={args.bf16}",
        f"--output_dir={args.output_dir}",
        f"--num_train_epochs={args.num_train_epochs}",
        f"--per_device_train_batch_size={args.per_device_train_batch_size}",
        f"--per_device_eval_batch_size={args.per_device_eval_batch_size}",
        f"--num_prompts={args.num_prompts}",
        f"--num_slots={args.num_slots}",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
        f"--save_strategy={args.save_strategy}",
        f"--save_steps={args.save_steps}",
        f"--save_total_limit={args.save_total_limit}",
        f"--learning_rate={args.learning_rate}",
        f"--weight_decay={args.weight_decay}",
        f"--warmup_ratio={args.warmup_ratio}",
        f"--lr_scheduler_type={args.lr_scheduler_type}",
        f"--logging_steps={args.logging_steps}",
        f"--tf32={args.tf32}",
        f"--model_max_length={args.model_max_length}",
        f"--gradient_checkpointing={args.gradient_checkpointing}",
        f"--lazy_preprocess={args.lazy_preprocess}",
        f"--report_to={args.report_to}",
        f"--lora_enable={args.lora_enable}",
        f"--freeze_backbone={args.freeze_backbone}",
        f"--freeze_prompt_router={args.freeze_prompt_router}"
    ]
)

from transllm.train.train_st_learning_prompt_5dataset import train

if __name__ == "__main__":
    train(model_args, data_args, training_args)
