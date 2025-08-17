import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from datasets import concatenate_datasets, load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import json
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
NEW_ADAPTER_NAME = "qwen3-4b-hp-assistant-adapter"
OASST_SAMPLES = 2000
ORCA_SAMPLES = 3000
# torch.cuda.set_device(2)
def create_formatted_dataset():
    """Loads, processes, merges, and robustly filters all datasets."""
    
    # 1. Define local paths
    LOCAL_DATA_PATH = "./datasets_local"
    oasst_path = os.path.join(LOCAL_DATA_PATH, "oasst1_subset")
    dolly_path = os.path.join(LOCAL_DATA_PATH, "dolly_15k")
    orca_math_path = os.path.join(LOCAL_DATA_PATH, "orca_math_subset")

    # 2. Load the datasets from disk
    print(f"1. Loading datasets from local disk: {LOCAL_DATA_PATH}...")
    oasst_dataset = load_from_disk(oasst_path)
    dolly_dataset = load_from_disk(dolly_path)
    orca_math_dataset = load_from_disk(orca_math_path)
    
    with open("hp_style_dataset.json", "r") as f:
        custom_data = json.load(f)

    # 3. Define tokenizer and preprocessing functions
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def process_oasst(example):
        # This function processes the oasst dataset.
        # It can return None if a row doesn't form a simple user/assistant chat.
        if example.get('parent_id') is None and example.get('text'):
            user_message = example['text']
            # Find the assistant's first reply to this message
            replies = [msg for msg in oasst_dataset if msg.get('parent_id') == example.get('message_id') and msg.get('role') == 'assistant']
            if replies:
                assistant_message = replies[0]['text']
                messages = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ]
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
        return {"text": None} # Return None in the 'text' field for invalid rows

    def process_dolly(example):
        instruction = example['instruction']
        context = example.get('context', '') # Use .get for safety
        response = example['response']
        user_message = f"{instruction}\n\n{context}".strip()
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    def process_custom(example):
        user_message = example['instruction']
        assistant_message = example['response']
        messages = [
            {"role": "system", "content": "You are a knowledgeable and helpful guide to the Wizarding World, akin to a friendly Hogwarts librarian."},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    
    def process_orca_math(example):
        user_message = example['question']
        assistant_message = example['answer']
        messages = [
            {"role": "system", "content": "You are a helpful assistant that thinks step-by-step."},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    # 4. Process and combine all datasets
    print("2. Processing and combining datasets...")
    
    processed_oasst = oasst_dataset.map(process_oasst, remove_columns=list(oasst_dataset.features))
    processed_dolly = dolly_dataset.map(process_dolly, remove_columns=list(dolly_dataset.features))
    processed_orca_math = orca_math_dataset.map(process_orca_math, remove_columns=list(orca_math_dataset.features))
    
    custom_dataset_hf = Dataset.from_list(custom_data)
    processed_custom = custom_dataset_hf.map(process_custom, remove_columns=list(custom_dataset_hf.features))
    
    combined_dataset = concatenate_datasets([
        processed_oasst, 
        processed_dolly, 
        processed_orca_math, 
        processed_custom
    ])

    final_dataset = combined_dataset.filter(
        lambda example: example.get("text") is not None and len(example["text"].strip()) > 0
    )
    
    print(f"Final combined and filtered dataset created with {len(final_dataset)} samples.")
    return final_dataset.shuffle(seed=42)



# --- MAIN SCRIPT ---

def main():
    print("🚀 Starting the fine-tuning process...")

    # 1. Create the formatted training dataset
    train_dataset = create_formatted_dataset()

    # 2. Configure 4-bit quantization
    print("3. Configuring quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # 3. Load the base model with quantization
    print(f"4. Loading base model: {BASE_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map= {"": torch.cuda.current_device()},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(model)

    # 4. Load the tokenizer
    print("5. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Configure LoRA
    print("6. Configuring LoRA...")
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adapter added to the model.")

    # 6. Configure Training Arguments
    print("7. Configuring training arguments with SFTConfig...")
    
    # SFTConfig combines TrainingArguments and SFT-specific arguments
    training_args = SFTConfig(
        # --- Standard Training Arguments ---
        output_dir="./results_hp_assistant",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        
        # --- SFT-specific Arguments (from the documentation) ---
        max_length=512,
        dataset_text_field="text", # This tells the trainer to use the 'text' column we created
        packing=True, # Packs multiple short sequences for efficiency
        ddp_find_unused_parameters=False,
    )

    # 7. Initialize the Trainer
    print("8. Initializing the SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args
    )


    # 8. Start Fine-Tuning
    print("\n" + "="*30)
    print("🚀 STARTING FINE-TUNING 🚀")
    print("="*30 + "\n")
    trainer.train()
    print("\n" + "="*30)
    print("✅ FINE-TUNING COMPLETE ✅")
    print("="*30 + "\n")

    # 9. Save the Fine-Tuned Adapter
    print(f"9. Saving adapter to '{NEW_ADAPTER_NAME}'...")
    trainer.model.save_pretrained(NEW_ADAPTER_NAME)
    print(f"Adapter saved successfully. You can now use '{NEW_ADAPTER_NAME}' for your RAG application.")


if __name__ == "__main__":
    main()