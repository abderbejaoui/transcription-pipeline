"""Test Qwen2.5-1.5B-Instruct on CPU — measure inference speed."""

import gc, sys, os, struct, json, time
gc.collect()
import torch
torch.cuda.empty_cache()

print(f"Torch: {torch.__version__}")
print(f"CPU cores available: {os.cpu_count()}")

# ─── Custom safetensors reader (CPU version) ────────────────

DTYPE_MAP = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

def load_safetensors(filename: str) -> dict:
    """Read a .safetensors file tensor-by-tensor on CPU. No mmap."""
    state_dict = {}
    with open(filename, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes)
        data_start = 8 + header_size
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_str = info["dtype"]
            shape = list(info["shape"])
            offsets = info["data_offsets"]

            f.seek(data_start + offsets[0])
            raw_bytes = f.read(offsets[1] - offsets[0])

            dtype = DTYPE_MAP.get(dtype_str, torch.float32)
            tensor = torch.frombuffer(raw_bytes, dtype=dtype).clone().reshape(shape)
            state_dict[name] = tensor

            if len(state_dict) % 50 == 0:
                print(f"  Loaded {len(state_dict)} tensors...")
    return state_dict

# ─── Find model files ──────────────────────────────────────

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
cache_dir = "D:/HF_CACHE"

snapshot_dir = None
base = os.path.join(cache_dir, "models--Qwen--Qwen2.5-1.5B-Instruct", "snapshots")
if os.path.isdir(base):
    snaps = os.listdir(base)
    if snaps:
        snapshot_dir = os.path.join(base, snaps[0])
if snapshot_dir is None:
    for root, dirs, _ in os.walk(os.path.join(cache_dir, "models--Qwen--Qwen2.5-1.5B-Instruct")):
        if os.path.basename(root) == "snapshots":
            snaps = os.listdir(root)
            if snaps:
                snapshot_dir = os.path.join(root, snaps[0])
                break

safetensor_files = sorted([
    os.path.join(snapshot_dir, f)
    for f in os.listdir(snapshot_dir)
    if f.endswith(".safetensors")
])
print(f"Safetensors files: {[os.path.basename(f) for f in safetensor_files]}")

# ─── Load model skeleton on CPU ─────────────────────────────

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

print("\nLoading config...")
config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
print(f"Type: {config.model_type}, hidden: {config.hidden_size}, layers: {config.num_hidden_layers}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)

print("Creating model skeleton on CPU...")
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
print(f"Skeleton created.")

# ─── Load weights one tensor at a time ──────────────────────

for sf in safetensor_files:
    fname = os.path.basename(sf)
    print(f"\nLoading {fname}...")
    state_dict = load_safetensors(filename=sf)
    print(f"  {len(state_dict)} tensors loaded. Loading state dict...")
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()

model.eval()
print(f"\nModel loaded on CPU.")

# ─── Test generation speed ──────────────────────────────────

print("\n=== Test generation ===")
messages = [
    {"role": "system", "content": "Identify misspelled medical terms. Respond ONLY with JSON."},
    {"role": "user", "content": "Sentence: Patient with myokardial infarction and prescribed metoprol. Identify misspelled medical terms."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")

print(f"Prompt length: {len(text)} chars, {inputs['input_ids'].shape[1]} tokens")
print("Generating...")
t0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
t1 = time.time()
resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
print(f"Response ({t1-t0:.1f}s): {resp}")

# Test 2 - more complex sentence
print("\n=== Test 2 ===")
messages2 = [
    {"role": "system", "content": "Identify misspelled medical terms. Respond ONLY with JSON."},
    {"role": "user", "content": "Sentence: The patient has fever, cough, and should take atorvasta and dolly prahn. Identify misspelled medical terms."},
]
text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
inputs2 = tokenizer([text2], return_tensors="pt")
print(f"Prompt: {text2[:200]}...")
t0 = time.time()
with torch.no_grad():
    outputs2 = model.generate(
        **inputs2,
        max_new_tokens=64,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
t1 = time.time()
resp2 = tokenizer.decode(outputs2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True).strip()
print(f"Response ({t1-t0:.1f}s): {resp2}")

print("\n✅ SUCCESS!")
