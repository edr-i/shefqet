import torch
import tiktoken
from src.model import GPTModel, generate
from src.weights import text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16
}

model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load("models/gpt2-medium355M-sft.pth", map_location=device))
model.eval()
model.to(device)

def chat(instruction, input_text=""):
    entry = {"instruction": instruction, "input": input_text}
    prompt = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
    )
    if input_text:
        prompt += f"\n\n### Input:\n{input_text}"
    prompt += "\n\n### Response:\n"

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    response = token_ids_to_text(token_ids, tokenizer)
    return response[len(prompt):].strip()

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
    print("Model:", chat(user_input))