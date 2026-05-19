from datasets import load_dataset
from trl import SFTTrainer,SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig

def format_prompts(example):
    eos = tokenizer.eos_token
    tool = example["assistant"]["tool_name"]

    text = f"""<|system|>
{example['system']}
<|user|>
{example['user']}
<|assistant|>
{{"tool_name": "{tool}"}}{eos}"""

    return text

raw_dataset = load_dataset("json", data_files="training_data.json", split="train")

def extract_tool(example):
    return {"target_tool": example["assistant"]["tool_name"]}

dataset = raw_dataset.map(extract_tool)

dataset = dataset.class_encode_column("target_tool")

train_test_split = dataset.train_test_split(test_size=0.3,stratify_by_column="target_tool",seed=42)

test_val_split = train_test_split["test"].train_test_split(test_size=0.5,stratify_by_column="target_tool",seed=42)

train_data = train_test_split["train"]
val_data = test_val_split["train"]
test_data = test_val_split["test"]

print(train_data["target_tool"])

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

from collections import Counter
print("Répartition dans le Train :", Counter(train_data["target_tool"]))
print("Répartition dans le Test :", Counter(test_data["target_tool"]))
print("Répartition dans le val :", Counter(val_data["target_tool"]))

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

sft_config = SFTConfig(
    max_length=2048,
    output_dir="./resultats",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    learning_rate=2e-5,
    label_smoothing_factor=0.1,

    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    max_grad_norm=1.0,
    weight_decay=0.01,

    num_train_epochs=6,

    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    load_best_model_at_end=True,
    save_total_limit=2,

    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=peft_config,
    args=sft_config,
    formatting_func=format_prompts,
    processing_class= tokenizer
)

trainer.train()