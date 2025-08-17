import os
import torch

print("\n--- GPU DIAGNOSTIC ---")

# 1. Check the environment variable as seen by your Python script
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible}")

# 2. Check what PyTorch can actually see
if not torch.cuda.is_available():
    print("❌ PyTorch cannot find any CUDA-enabled GPUs.")
else:
    device_count = torch.cuda.device_count()
    print(f"✅ PyTorch sees {device_count} GPU(s).")
    
    if device_count > 0:
        current_device_id = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device_id)
        print(f"   - Current device ID is: {current_device_id}")
        print(f"   - Current device name is: {current_device_name}")

print("--- END DIAGNOSTIC ---\n")