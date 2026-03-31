import logging
import os
import time

from mlx_lm import generate, load

logging.basicConfig(level=logging.INFO)
os.environ["TQ_USE_METAL"] = "1"

model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
print(f"Loading {model_name}...")
model, tokenizer = load(model_name)

prompt = "Write a haiku about programming in C and Metal."
if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

print("\n--- Generating with TurboQuant (Metal Native) ---")
start_time = time.time()
response = generate(
    model,
    tokenizer,
    prompt,
    max_tokens=200,
    verbose=True,
    turboquant_k_start=0,
    turboquant_main_bits=3,
    turboquant_group_size=64,
)
end_time = time.time()
print(f"\nTime taken: {end_time - start_time:.2f}s")
