import os
os.environ["HF_HOME"] = "/home/zzz/.cache/huggingface"
# from transformers import AutoTokenizer
from modelscope.utils.hf_util import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

dataset = load_dataset("YeungNLP/firefly-train-1.1M", split="train[:500]") 


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.padding_size = "left"

def format_prompt(example):
    chat = [
        {"role": "system", "content": "你是一个非常棒的人工智能助手"},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["target"]}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}

dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

print(dataset[10])

## load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")

## LoRA Config 

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'v_proj', 'q_proj']
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

from transformers import TrainingArguments
output_dir = "./results"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True,
    
    save_steps=15,
    max_steps=20,
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    dataset_text_field="text",
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_seq_length=512
)

trainer.train()

trainer.model.save_pretrained("./result/final-result")
