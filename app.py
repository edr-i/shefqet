import tiktoken
import torch
import chainlit
from pathlib import Path
from src.model import GPTModel, generate_streaming
from src.weights import text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_and_tokenizer():
    GPT_CONFIG_355M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    tokenizer = tiktoken.get_encoding("gpt2")
    model_path = Path("models") / "gpt2-medium355M-sft.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Run finetune.py first."
        )
    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    return tokenizer, model, GPT_CONFIG_355M

tokenizer, model, model_config = get_model_and_tokenizer()

@chainlit.on_chat_start
async def start():
    await chainlit.Message(
        content="Hi! I'm a GPT-2 model fine-tuned on instruction data. Ask me anything."
    ).send()

@chainlit.on_message
async def main(message: chainlit.Message):
    torch.manual_seed(123)
    prompt = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{message.content}"
        f"\n\n### Response:\n"
    )

    input_ids = text_to_token_ids(prompt, tokenizer).to(device)

    # create an empty message to stream into
    response_message = chainlit.Message(content="")
    await response_message.send()

    for token_id in generate_streaming(
        model=model,
        idx=input_ids,
        max_new_tokens=256,
        context_size=model_config["context_length"],
        eos_id=50256
    ):
        token_text = tokenizer.decode([token_id])
        await response_message.stream_token(token_text)

    await response_message.update()