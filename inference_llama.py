import os
import sys
import json
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, concatenate_datasets
import transformers
import argparse
import warnings
from huggingface_hub import snapshot_download
from transformers import EarlyStoppingCallback, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

pid = os.getpid()
print('pid: ', pid)

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--output_path", type=str, default="lora-Vicuna")
parser.add_argument("--model_path", type=str)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--train_size", type=int, default=512)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup_ratio", type=float, default=0.0)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--use_lora", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--dataset", type=str, default='BookCrossing')
parser.add_argument("--dataset_dir", type=str, default="../ReLLa/data")
parser.add_argument("--neftune_noise_alpha", type=float, default=-1)

# Here are args of prompt
#recent and random needs 'K'
parser.add_argument("--K", type=int, default=15)
parser.add_argument("--test_size", type=float, default=1.0, help='sample a sub-set as the test set')
parser.add_argument("--exp_name", type=str, default='')
parser.add_argument("--random_test", default=False)
#TPMI params
parser.add_argument("--alpha", type=float, default=0.05)
# dynamic parames
parser.add_argument("--p", type=float, default=1.5)
#
parser.add_argument("--temp_type", type=str, default="dynamic")

args = parser.parse_args()

data_path = os.path.join(args.dataset_dir, args.dataset, 'proc_data/data')

# Fit for single card V100, increasing bs if GPU allows is OK.
if args.temp_type in ['mixed', 'recent', 'random']:
    # Fit for single card V100, increasing bs if GPU allows is OK.
    if args.K <= 15:
        args.per_device_eval_batch_size = 8
    elif args.K <= 40:
        args.per_device_eval_batch_size = 2
    else:
        args.per_device_eval_batch_size = 2
else:
    #dynamic allow bigger bs
    args.per_device_eval_batch_size = 10


print('*'*100)
print(args)
print('*'*100)

transformers.set_seed(args.seed)

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"


MICRO_BATCH_SIZE = args.per_device_eval_batch_size
BATCH_SIZE = min(args.total_batch_size, args.per_device_eval_batch_size)
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs
LEARNING_RATE = args.lr
CUTOFF_LEN = 2048
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.val_size #2000
USE_8bit = True
FP16 = False
OUTPUT_DIR = args.output_path
Eval_DEVIATION = 1
        
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]



if args.temp_type == 'dynamic':
    #load sorted df
    fp_test = f"test_{args.temp_type}_p{args.p}_tpmi_alpha{args.alpha}.parquet.gz"
    DATA_PATH = {
        "test": '/'.join([data_path, f"test/test_dynamic_p{args.p}_tpmi_alpha{args.alpha}.json"])
    }
else:
    fp_test = f"test.parquet.gz" 
    DATA_PATH = {
        "test": '/'.join([data_path, f"test/test_K{args.K}_{args.temp_type}.json"])
    }

data_dir = os.path.join(args.dataset_dir, f'{args.dataset}/proc_data')
test = pd.read_parquet(os.path.join(data_dir, fp_test))
print(f"DATA_PATH: {DATA_PATH}")


device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
print(args.model_path)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    use_fast=False,
    add_eos_token=True, 
)

if USE_8bit is True:
    print('Use 8bit!')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules= ['lm_head'],
        ),
    )
    model = prepare_model_for_kbit_training(model)
else:
    FP16 = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

if args.use_lora:
    model = PeftModel.from_pretrained(
    model, 
    args.output_path, 
    is_trainable=True, 
    )
else:
    setattr(model, "_hf_peft_config_loaded", True) # hack here: make model compatible with prediction
#     config = LoraConfig(
#         r=LORA_R,
#         lora_alpha=LORA_ALPHA,
#         target_modules=TARGET_MODULES,
#         lora_dropout=LORA_DROPOUT,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )
#     model = get_peft_model(model, config)
#     print("Lora used.")

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
# tokenizer.padding_side = "left"  # Allow batched inference

data = load_dataset("json", data_files=DATA_PATH)

if args.test_size < 1.0:
    test_num = int(args.test_size*len(data['test']))
    print(f"Use a sub-set testset, test num: {test_num}")
    data["test"] = data["test"].select(range(test_num))
    test = test.iloc[:test_num, :]
else:
    test_num = len(data['test'])
print(f'test data num: {test_num}')
print("Data loaded.")

now_max_steps = 100
if args.resume_from_checkpoint:
    if args.lora_remote_checkpoint is not None:
        snapshot_download(repo_id=args.lora_remote_checkpoint, allow_patterns=["*.pt", "*.bin", "*.json"], local_dir=args.resume_from_checkpoint)
    checkpoint_name = os.path.join(
        args.resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            warnings.warn("The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = (
                None
            )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        model = set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")
    
    train_args_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
    
    if os.path.exists(train_args_path):
        import json
        base_train_args = json.load(open(train_args_path, 'r'))
        base_max_steps = base_train_args["max_steps"]
        resume_scale = base_max_steps / now_max_steps
        if base_max_steps > now_max_steps:
            warnings.warn("epoch {} replace to the base_max_steps {}".format(EPOCHS, base_max_steps))
            EPOCHS = None
            MAX_STEPS = base_max_steps
        else:
            MAX_STEPS = now_max_steps
else:
    MAX_STEPS = now_max_steps


def generate_and_tokenize_prompt(data_point):
    messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {"role": "user", "content": data_point["input"]},
     ]
    user_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
            )["input_ids"]
        )
    )
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
            )["input_ids"]
        ) -1
    ) -1
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        # padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
#         "attention_mask": [1] * (len(full_tokens)),
    }


test_data = data['test'].map(generate_and_tokenize_prompt)
print("Data processed.")


def compute_metrics(eval_preds):
    global curr_best
    pre, labels = eval_preds
    test['predictions'] = pre[0]
    auc = roc_auc_score(pre[1], pre[0])
    ll = log_loss(pre[1], pre[0])
    acc = accuracy_score(pre[1], pre[0] > 0.5)

    res_dir = f'./results/{args.dataset}_{args.exp_name}.parquet.gz'
    test.to_parquet(res_dir, compression="gzip")
    print(f'Test Results Save at: ', res_dir)
        
    return {
        'auc': auc, 
        'll': ll, 
        'acc': acc, 
    }


#llama
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    labels: (N, seq_len), logits: (N, seq_len, 32000)
    """
    labels_index = torch.argwhere(torch.bitwise_or(labels == 9642, labels == 2822))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 2822, 0, 1)
    labels_index[: , 1] = labels_index[: , 1] - 1
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [2822, 9642]]
    prob = torch.softmax(logits, dim=-1)
    return prob[:, 1], gold


dataset_size = len(test_data)  # Number of examples in the training dataset
batch_size = MICRO_BATCH_SIZE  # Batch size per device (GPU)
gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS  # Gradient accumulation steps

# Calculate total number of steps per epoch and per half epoch
# steps_per_epoch = math.ceil(dataset_size / (batch_size * gradient_accumulation_steps))
# steps_per_half_epoch = steps_per_epoch // Eval_DEVIATION
# print(f'Evaluation Steps: {steps_per_half_epoch}')

trainer = transformers.Trainer(
    model=model,
    train_dataset=test_data,
    eval_dataset=test_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        # max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=FP16,
        logging_strategy="steps",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
#         eval_steps=args.eval_steps,
#         save_steps=args.save_steps,
        # eval_strategy="steps",
        # save_strategy="steps",
        # eval_steps=steps_per_half_epoch,
        # save_steps=steps_per_half_epoch,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_auc",
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
        warmup_ratio=args.warmup_ratio,
        neftune_noise_alpha=args.neftune_noise_alpha if args.neftune_noise_alpha > 0 else None,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding='longest'
    ),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
)
model.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

print("Start evaluation...")
print(trainer.evaluate())

