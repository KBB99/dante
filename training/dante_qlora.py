from typing import Dict

from torch import Tensor
from transformers import LlamaForCausalLM, AutoConfig, AutoModelForCausalLM
import torch
import os
import dotenv
from models.barktok.modeling_barktok import BarkTok
from trainers.base import BaseModule, BaseEvaluation
from dataset.preprocessed import PreProcessedDataset

import os
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

dotenv.load_dotenv()



# A training script for Llama that uses multi gpu training and PEFT for qlora



# load llama in flash attn mode
model_name = "meta-llama/Llama-2-7b-hf"

# quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #device_map="auto",
    device_map={"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))},
    #config=config,
    trust_remote_code=True,
    quantization_config=bnb_config,
    use_auth_token=os.environ['HUGGINGFACE_TOKEN'],
)


model = prepare_model_for_kbit_training(model)

# lora config, getting all module names, attention and linears
target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

import re
model_modules = str(model.modules)
pattern = r'\((\w+)\): Linear'
linear_layer_names = re.findall(pattern, model_modules)

names = []
for name in linear_layer_names:
    names.append(name)
target_modules.extend(list(set(names)))

lora_config = LoraConfig(
    r=32,
    target_modules = target_modules,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#tokenizer: BarkTok = BarkTok()

dataset = PreProcessedDataset(
    dataset_name="dante-podcasts-processed",
    dataset_length=120000,
    account_name="audiogentrainingdataeun"
)

training_args = transformers.TrainingArguments(
    #hub_model_id="Audiogen/dante-llama13b-qlora-hackathon",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=8,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    #auto_find_batch_size=True,
    gradient_accumulation_steps=1,
    num_train_epochs=100000,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=4,
    logging_steps=5,
    output_dir='~/training/logs',
    save_strategy='steps',
    save_steps=70,
    evaluation_strategy="steps",
    eval_steps=70,
    run_name="dante",
    #push_to_hub=True,
    #hub_token=os.environ['HUGGINGFACE_TOKEN'],
    torch_compile=False,
)


class MyTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)
    
# split dataset
len_dataset = len(dataset)
split = 0.005
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [int(len_dataset * (1 - split)), int(len_dataset * split)])

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


if __name__ == "__main__":
    trainer.train()
