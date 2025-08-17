import torch
from datasets import load_dataset
from peft import PeftModel, LoraConfig, TaskType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import RewardConfig, RewardTrainer
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "./qwen3-4b-hp-assistant-adapter"
PREFERENCE_DATASET_PATH = "feedback_log.jsonl"
REWARD_MODEL_NAME = "qwen3-4b-hp-reward-model"
TEMP_MERGED_MODEL_PATH = "./temp_merged_sft_model"

# --- 1. Load, Merge, and Save the SFT Model ---
print("Loading SFT model, merging adapter, and saving temporarily...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()},
    trust_remote_code=True,
)
sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
merged_sft_model = sft_model.merge_and_unload()

merged_sft_model.save_pretrained(TEMP_MERGED_MODEL_PATH)
tokenizer.save_pretrained(TEMP_MERGED_MODEL_PATH)

# --- 2. Load the Merged Model as a Sequence Classification Model ---
print(f"Loading merged model from '{TEMP_MERGED_MODEL_PATH}' as a Reward Model...")

model = AutoModelForSequenceClassification.from_pretrained(
    TEMP_MERGED_MODEL_PATH,
    num_labels=1,
    device_map={"": torch.cuda.current_device()},
    trust_remote_code=True,
    # quantization_config=bnb_config, <-- THIS LINE IS REMOVED TO FIX THE WARNING
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# --- 3. Load and Format the Preference Dataset ---
print(f"Loading preference dataset from {PREFERENCE_DATASET_PATH}...")
train_dataset = load_dataset("json", data_files=PREFERENCE_DATASET_PATH, split="train")

def format_for_reward_modeling(example):
    chosen_full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": example["chosen"]}],
        tokenize=False
    )
    rejected_full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": example["rejected"]}],
        tokenize=False
    )
    return {"chosen": chosen_full_text, "rejected": rejected_full_text}

formatted_dataset = train_dataset.map(format_for_reward_modeling)

# --- 4. Define PEFT Config for Reward Model ---
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

# --- 5. Train the Reward Model ---
training_args = RewardConfig(
    output_dir="./reward_model_results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    report_to="tensorboard",
    logging_steps=10,
    save_steps=50,
    max_length=512,
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
)

print("Starting Reward Model training...")
trainer.train()

print(f"Saving Reward Model to {REWARD_MODEL_NAME}...")
trainer.save_model(REWARD_MODEL_NAME)
tokenizer.save_pretrained(REWARD_MODEL_NAME)

print("✅ Reward Model training complete!")