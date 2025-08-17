import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from tqdm import tqdm

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "./qwen3-4b-hp-assistant-adapter"
REWARD_MODEL_PATH = "./qwen3-4b-hp-reward-model"
FINAL_RLHF_ADAPTER_NAME = "qwen3-4b-hp-rlhf-adapter"
PROMPT_DATASET_PATH = "feedback_log.jsonl"

# --- 1. PPO Configuration ---
config = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    report_to="tensorboard",
    num_ppo_epochs=4,
)

# --- 2. Load Models and Tokenizer ---
print("Loading models for PPO training...")

tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True,
    device_map={"": torch.cuda.current_device()},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
)

sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)

policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    sft_model,
    peft_config=lora_config,
    trust_remote_code=True,
)

# --- THIS IS THE FIX ---
# CORRECTED CODE
policy_model.base_model_prefix = "base_model"
policy_model.model.base_model_prefix = "model" # Use .model instead of .base_model
# --- END OF FIX ---

policy_model.generation_config = sft_model.generation_config

ref_model = create_reference_model(policy_model)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH,
    num_labels=1,
    trust_remote_code=True,
    device_map={"": torch.cuda.current_device()},
)
print("Models loaded successfully.")

# --- 3. Prepare the Dataset ---
def build_dataset(tokenizer, dataset_path):
    def tokenize(example):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        tokenized = tokenizer(formatted_prompt, truncation=True)
        tokenized["query"] = formatted_prompt
        return tokenized

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(tokenize, remove_columns=list(dataset.features))
    dataset.set_format("torch")
    return dataset

dataset = build_dataset(tokenizer, PROMPT_DATASET_PATH)

# --- 4. Initialize PPOTrainer ---
ppo_trainer = PPOTrainer(
    args=config,
    model=policy_model,
    ref_model=ref_model,
    processing_class=tokenizer,
    train_dataset=dataset,
    reward_model=reward_model,
    value_model=policy_model,
)

# --- 5. The PPO Training Loop ---
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 128,
}

print("Starting PPO training...")
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    texts_for_reward = [q + r for q, r in zip(batch["query"], batch["response"])]
    reward_inputs = tokenizer(texts_for_reward, padding=True, truncation=True, return_tensors="pt").to(ppo_trainer.accelerator.device)
    
    with torch.no_grad():
        scores = reward_model(**reward_inputs).logits
    
    rewards = [score.detach() for score in scores]

    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

# --- 6. Save the final model ---
print(f"Saving final RLHF adapter to {FINAL_RLHF_ADAPTER_NAME}...")
ppo_trainer.model.save_pretrained(FINAL_RLHF_ADAPTER_NAME)
print("✅ RLHF (PPO) training complete!")