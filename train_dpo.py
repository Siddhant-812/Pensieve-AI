import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "./qwen3-4-hp-assistant-adapter"
PREFERENCE_DATASET_PATH = "./preference_data.jsonl"
FINAL_DPO_ADAPTER_NAME = "qwen3-4b-hp-dpo-adapter"

# --- 1. Load SFT Model and Tokenizer ---
print("Loading SFT-tuned model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()},
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the SFT adapter onto the base model
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)

# --- 2. Load Preference Dataset ---
print(f"Loading preference dataset from {PREFERENCE_DATASET_PATH}...")
train_dataset = load_dataset("json", data_files=PREFERENCE_DATASET_PATH, split="train")

# --- 3. DPO Training ---
print("Configuring DPO training...")

training_args = DPOConfig(
    output_dir="./dpo_results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    beta=0.1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    report_to="tensorboard",
    logging_steps=10,
    save_steps=100,
    max_length=1024, # Max length of prompt + response
    max_prompt_length=512, # Max length of prompt
)

# Initialize the DPOTrainer
dpo_trainer = DPOTrainer(
    model,
    ref_model=None, # The trainer will handle the reference model automatically
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

print("Starting DPO training...")
dpo_trainer.train()

print(f"Saving DPO-tuned adapter to {FINAL_DPO_ADAPTER_NAME}...")
dpo_trainer.model.save_pretrained(FINAL_DPO_ADAPTER_NAME)

print("✅ DPO training complete!")