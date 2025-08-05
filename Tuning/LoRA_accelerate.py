# !pip install datasets evaluate accelerate rouge_score transformers peft
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

model_name = "Qwen/Qwen3-4B"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='cuda'
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_name,
    trust_remote_code=True
)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model=model, peft_config=lora_cfg)
model.print_trainable_parameters()

dataset = load_dataset("tatsu-lab/alpaca", split="train[:400]")

def tokenize(example):
    prompt = example["instruction"]
    completion = example["output"]
    text = example["text"]

    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512
    )

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=512
    )["input_ids"]

    prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    labels = labels[:512]
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type="torch")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

training_args = TrainingArguments(
    output_dir="./qwen3-4b-lora-output",
    per_device_train_batch_size=6,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    bf16=False,
    report_to="none",
)

from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW

from tqdm import tqdm

accelerator = Accelerator()
accelerator.print(f"Using {accelerator.num_processes} processes on device {accelerator.device}")
train_dataloader = DataLoader(
    tokenized_dataset,
    shuffle=True,
    batch_size=training_args.per_device_train_batch_size,
    collate_fn=data_collator
)

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * training_args.num_train_epochs,
)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)
print('[DEBUG] Start Training!')
model.train()
for epoch in range(training_args.num_train_epochs):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.set_postfix(loss=loss.item())

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./qwen3-4b-lora-adapter", save_function=accelerator.save)
tokenizer.save_pretrained("./qwen3-4b-lora-adapter")


# ================================= Evaluation ======================================

import evaluate

eval_dataset = load_dataset("tatsu-lab/alpaca", split="train[2000:2010]")

rouge = evaluate.load("rouge")

predictions = []
references = []

model.eval()

for sample in eval_dataset:
    instruction = sample["instruction"]
    input_text = sample.get("input", "")
    prompt = instruction if not input_text else instruction + "\n" + input_text

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated = decoded[len(prompt):].strip()
    predictions.append(generated)
    references.append(sample["output"].strip())

results = rouge.compute(predictions=predictions, references=references)

print("\nROUGE Evaluation Results:")
for key, score in results.items():
    print(f"{key}: {score:.4f}")


import json
import os
from datetime import datetime

log_dir = "./eval_logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(log_dir, f"rouge_eval_{timestamp}.json")

with open(log_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n ROUGE 评估结果已保存至: {log_path}")
