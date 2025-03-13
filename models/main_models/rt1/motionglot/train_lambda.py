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
import random 
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import TrainerCallback

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer 

def GPT_model(freeze_all=False, unfreeze_head=False):
    
    config = GPT2Config.from_pretrained(args.pre_train_model)

    # config.attn_pdrop = 0.2  # Change attention dropout to 0.3
    # config.resid_pdrop = 0.2  # Change residual dropout to 0.3
    # config.embd_pdrop = 0.2   # Change embedding dropout to 0.3

    # model = GPT2LMHeadModel.from_pretrained(args.pre_train_model  ).to(args.device)

    model = GPT2LMHeadModel.from_pretrained(args.pre_train_model , state_dict=None)
    model.init_weights()
    model.resize_token_embeddings( len(tokenizer) )

    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False
        if unfreeze_head:
            for param in model.lm_head.parameters():
                param.requires_grad = True

    return model 

def process_data():
    
    train_dataset = Dataset.from_dict({"input_ids": DATA['train_data'].tolist()  , "attention_mask" : DATA['train_mask'].tolist() , "labels" : DATA['train_data'].clone().tolist() } )
    valid_dataset = Dataset.from_dict({"input_ids": DATA['valid_data' ][0:100, : ].tolist()  , "attention_mask" : DATA['valid_mask'][0:100, : ].tolist() , "labels" : DATA['valid_data'][0:100, : ].clone().tolist()  })

    tokenized_datasets = DatasetDict({"train": train_dataset, "valid":valid_dataset})

    return tokenized_datasets 

def trainer_args():
    
    return TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    evaluation_strategy="steps",  # Evaluate regularly during training.
    eval_steps=50,  # Evaluate every 50 steps.
    logging_steps=1,
    gradient_accumulation_steps=2,
    num_train_epochs=600,
    weight_decay=0.1,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=100,  # Save checkpoints every 100 steps.
    save_strategy="steps",  # Save strategy for regular checkpoints.
    save_total_limit=5,
    metric_for_best_model="eval_loss",  # Metric to determine the best model.
    greater_is_better=False,  # Lower eval_loss is better.
    load_best_model_at_end=True,  # Ensure best model is loaded at the end.
    push_to_hub=False,
    report_to="wandb"
)


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, trainer, metric_name="eval_loss", greater_is_better=False):
        self.trainer = trainer
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics.get(self.metric_name)
        if current_metric is not None:
            if (
                self.best_metric is None or
                (self.greater_is_better and current_metric > self.best_metric) or
                (not self.greater_is_better and current_metric < self.best_metric)
            ):
                self.best_metric = current_metric
                print(f"New best model found: {self.metric_name} = {self.best_metric}")
                
                # Save model
                self.trainer.save_model(output_dir=f"{args.output_dir}/checkpoint-best")
                
                # Save trainer state
                self.trainer.state.save_to_json(os.path.join(f"{args.output_dir}/checkpoint-best", "trainer_state.json"))
                
                # Save optimizer state
                torch.save(self.trainer.optimizer.state_dict(), os.path.join(f"{args.output_dir}/checkpoint-best", "optimizer.pt"))
                
                # Save scheduler state
                torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(f"{args.output_dir}/checkpoint-best", "scheduler.pt"))
                
                # Save RNG state
                torch.save(torch.get_rng_state(), os.path.join(f"{args.output_dir}/checkpoint-best", "rng_state.pth"))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Train MotionGlot ")
    parser.add_argument("--tokenizer_path", help=" path to folder with tokenizer " , default= "lambda_tokenizer/lambda_task_gen", type= str ) 
    parser.add_argument("--dataset", help=" path to dataset " , default= "tokenized_dataset_pickles/lambda_task_gen.pkl", type= str ) 
    parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    parser.add_argument("--output_dir", help=" output dir  " , default= "", type= str )
    parser.add_argument("--pre_train_model", help=" set path to pre train model " , default= "gpt2" , type= str )
    parser.add_argument("--freeze_all", action="store_true", help="freeze the gpt-2 model")
    parser.add_argument("--unfreeze_head", action="store_true", help="unfreeze the gpt-2 head")

    args = parser.parse_args()

    print(args)

    with open( args.dataset, "rb") as f:
        DATA = pickle.load(f)
    
    tokenizer = get_tokenizer()

    GPT = GPT_model(args.freeze_all, args.unfreeze_head)
    tokenized_datset =  process_data()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False )

    train_args = trainer_args()

    # Initialize the Trainer
    trainer = Trainer(
        model=GPT,  
        args=train_args,
        data_collator=data_collator,
        train_dataset=tokenized_datset["train"], 
        eval_dataset=tokenized_datset["valid"],  
    )

    # Attach the SaveBestModelCallback
    save_best_callback = SaveBestModelCallback(
        trainer=trainer,
        metric_name="eval_loss",  # Replace with the metric you're tracking (e.g., accuracy).
        greater_is_better=False  # Use True if higher metric values are better.
    )

    # Add the callback to the trainer
    trainer.add_callback(save_best_callback)

    torch.cuda.empty_cache()

    trainer.train()