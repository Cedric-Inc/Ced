
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

dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")

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
    per_device_train_batch_size=2,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./qwen3-4b-lora-adapter")
tokenizer.save_pretrained("./qwen3-4b-lora-adapter")
