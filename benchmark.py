import os
import time
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Device agnostic runtime assignment
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Initializing benchmarking framework on execution context: {device.upper()}")

def compute_sequence_entropy(outputs):
    """Calculates token-level Shannon Entropy as an uncertainty proxy."""
    if not hasattr(outputs, "scores") or len(outputs.scores) == 0:
        return 0.0
    sequence_entropy = []
    for logit_tensor in outputs.scores:
        probs = torch.softmax(logit_tensor, dim=-1)
        non_zero_probs = probs[probs > 0]
        entropy = -torch.sum(non_zero_probs * torch.log(non_zero_probs)).item()
        sequence_entropy.append(entropy)
    return np.mean(sequence_entropy)

def profile_configuration(model_id, config_name="uncompressed"):
    """Profiles a specific transformer precision layout structure."""
    print(f"\n[+] Executing profiling matrix for layout: {config_name.upper()}")
    
    # Configure parameter-efficient quantization flags if hardware supports it
    if config_name == "4bit" and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    elif config_name == "8bit" and device == "cuda":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None
        if config_name in ["4bit", "8bit"]:
            print(f"    [!] VRAM quantization skipped: Not supported on target device engine ({device.upper()})")

    # Load tokenizer and model parameters dynamically
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        
    model.eval()

    # Standardization evaluation context matching systems engineering constraints
    prompt = "Systems engineering solutions for sustainable digital sovereignty require"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids, max_new_tokens=30, return_dict_in_generate=True, output_scores=True, do_sample=False
        )
    latency = time.time() - start_time
    generated_tokens_count = len(outputs.scores)
    tokens_per_second = generated_tokens_count / latency
    mean_entropy = compute_sequence_entropy(outputs)
    
    # VRAM metric extraction fallback logic
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = (model.num_parameters() * 4) / (1024 ** 2) # Theoretical FP32 baseline estimate

    return {
        "Compression Mode": config_name.upper(),
        "Compute Latency (s)": round(latency, 4),
        "Generation Velocity (tok/s)": round(tokens_per_second, 2),
        "Predictive Sequence Entropy": round(mean_entropy, 4),
        "Peak VRAM Allocation (MB)": round(peak_memory_mb, 2)
    }

if __name__ == "__main__":
    TARGET_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    evaluation_matrix = ["uncompressed", "8bit", "4bit"] if device == "cuda" else ["uncompressed"]
    
    benchmarking_results = []
    for mode in evaluation_matrix:
        try:
            metrics = profile_configuration(TARGET_MODEL, config_name=mode)
            benchmarking_results.append(metrics)
        except Exception as e:
            print(f"[-] Execution layout failure for {mode}: {e}")

    if benchmarking_results:
        df = pd.DataFrame(benchmarking_results)
        print("\n" + "="*80 + "\n          EMPIRICAL PERFORMANCE MATRIX GENERATED SUCCESSFULLY\n" + "="*80)
        print(tabulate(df, headers='keys', tablefmt='markdown', showindex=False))
        print("="*80)
