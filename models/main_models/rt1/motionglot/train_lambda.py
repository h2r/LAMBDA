import os
import torch
import zipfile
from tokenizers import (decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer)
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel , GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import logging
import time
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments
import argparse
import pickle 
from transformers import Trainer, TrainingArguments
import random 
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer 

def GPT_model():
    
    config = GPT2Config.from_pretrained(args.pre_train_model)

    # config.attn_pdrop = 0.2  # Change attention dropout to 0.3
    # config.resid_pdrop = 0.2  # Change residual dropout to 0.3
    # config.embd_pdrop = 0.2   # Change embedding dropout to 0.3

    # model = GPT2LMHeadModel.from_pretrained(args.pre_train_model  ).to(args.device)

    model = GPT2LMHeadModel.from_pretrained(args.pre_train_model , state_dict=None)
    model.init_weights()
    model.resize_token_embeddings( len(tokenizer) )
    
    return model 

def process_data():
    
    train_dataset = Dataset.from_dict({"input_ids": DATA['train_data'].tolist()  , "attention_mask" : DATA['train_mask'].tolist() , "labels" : DATA['train_data'].clone().tolist() } )
    valid_dataset = Dataset.from_dict({"input_ids": DATA['valid_data' ][0:100, : ].tolist()  , "attention_mask" : DATA['valid_mask'][0:100, : ].tolist() , "labels" : DATA['valid_data'][0:100, : ].clone().tolist()  })

    tokenized_datasets = DatasetDict({"train": train_dataset, "valid":valid_dataset})

    return tokenized_datasets 

def trainer_args():
    
    return TrainingArguments(
    output_dir="pick_place_manip",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_steps=1,
    gradient_accumulation_steps=2,
    num_train_epochs=600, #38
    weight_decay=0.1,
    warmup_steps=500, 
    lr_scheduler_type="cosine",
    learning_rate= 5e-4, ## decrase this to 2e-4 
    save_steps=100,
    push_to_hub=False,
    save_total_limit=8,
    report_to ="wandb"
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Train MotionGlot ")
    parser.add_argument("--tokenizer_path", help=" path to folder with tokenizer " , default= "/oscar/scratch/sharitha/research/motionglot/tokenizer/ImageTokenizer/lambda_tokenizer/moma", type= str ) 
    parser.add_argument("--dataset", help=" path to dataset " , default= "/oscar/scratch/sharitha/research/motionglot/tokenizer/ImageTokenizer/train_data_lambda/moma.pkl", type= str ) 
    parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    parser.add_argument("--pre_train_model", help=" set path to pre train model " , default= "gpt2" , type= str )

    args = parser.parse_args()

    print(args)

    with open( args.dataset, "rb") as f:
        DATA = pickle.load(f)
    
    tokenizer = get_tokenizer()

    GPT = GPT_model()
    tokenized_datset =  process_data()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False )

    train_args = trainer_args()

    trainer = Trainer(
        model=GPT,
        args=train_args,
        data_collator=data_collator,
        train_dataset=tokenized_datset["train"],
        eval_dataset=tokenized_datset["valid"],
    )

    trainer.train()