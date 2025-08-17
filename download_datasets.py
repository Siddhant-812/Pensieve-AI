from datasets import load_dataset
import os

# --- Configuration ---
SAMPLES_TO_SAVE = {
    "oasst1": 2000,
    "orca_math": 10000  # Adding 10k samples for CoT
}
SAVE_PATH = "./datasets_local"
os.makedirs(SAVE_PATH, exist_ok=True)

# --- Download and Save oasst1 ---
print(f"Downloading {SAMPLES_TO_SAVE['oasst1']} samples from OpenAssistant/oasst1...")
oasst_subset = load_dataset("OpenAssistant/oasst1", split=f"train[:{SAMPLES_TO_SAVE['oasst1']}]")
oasst_save_path = os.path.join(SAVE_PATH, "oasst1_subset")
oasst_subset.save_to_disk(oasst_save_path)
print("... oasst1 subset saved.\n")

# --- Download and Save dolly-15k ---
print("Downloading databricks/databricks-dolly-15k...")
dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
dolly_save_path = os.path.join(SAVE_PATH, "dolly_15k")
dolly_dataset.save_to_disk(dolly_save_path)
print("... dolly-15k saved.\n")

# --- Download and Save orca-math-word-problems ---
print(f"Downloading {SAMPLES_TO_SAVE['orca_math']} samples from microsoft/orca-math-word-problems-200k...")
orca_math_subset = load_dataset("microsoft/orca-math-word-problems-200k", split=f"train[:{SAMPLES_TO_SAVE['orca_math']}]")
orca_math_save_path = os.path.join(SAVE_PATH, "orca_math_subset")
orca_math_subset.save_to_disk(orca_math_save_path)
print("... orca-math subset saved.\n")


print("✅ All datasets have been downloaded and saved locally.")