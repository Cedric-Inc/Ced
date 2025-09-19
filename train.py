import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import PretrainedConfig, PreTrainedModel
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.optim import AdamW
from transformers.modeling_outputs import CausalLMOutput
from modeling_cedric import CedricForCausalLM, args

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("dhruveshpatel/tiny_roc_stories")
# , split=["train[:10000]", "validation[:1000]"]


def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=args.max_seq_len,
    )


tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text", "source"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

model = CedricForCausalLM(args)

training_args = TrainingArguments(
    # output_dir="/content/drive/MyDrive/LLM/outputs_d",
    output_dir="./outputs",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=400,
    save_strategy="steps",
    report_to="none",
    bf16=False,
    deepspeed="ds_config.json",
    # evaluate_during_training=True,
    eval_strategy="steps",
    load_best_model_at_end=True,
    # metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

import time

t0 = time.time()

trainer.train()
print('cost:', time.time() - t0, 's')

trainer.save_model("./outputs_d/Ced_33M")
tokenizer.save_pretrained("./outputs_d/Ced_33M")

from huggingface_hub import login

login("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")

model.push_to_hub("UserCedric/Ced-33M", overwrite=True)
tokenizer.push_to_hub("UserCedric/Ced-33M", overwrite=True)

