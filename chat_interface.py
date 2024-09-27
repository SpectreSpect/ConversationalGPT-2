from src.models.conversational_gpt2.conversational_gpt2 import ConvGPT2, TextCollectorCallback, PrintTokensCallback
import torch
from dataclasses import dataclass
import tiktoken
import kagglehub
import os
import shutil

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


def get_str_slice(array: list, slice):
    start = slice[0]
    stop = start = slice[1]

    start_string_id = -1
    start_id_in_string = -1
    stop_string_id = -1
    stop_id_in_string = -1

    num_passed_chars = 0
    for idx, string in enumerate(array):
        if start_string_id < 0 and num_passed_chars + len(string) > start:
            start_string_id = idx
            start_id_in_string = num_passed_chars - start
        if stop_string_id < 0 and num_passed_chars + len(string) > stop:
            stop_string_id = idx
            stop_id_in_string = num_passed_chars - stop
        
        num_passed_chars += len(string)

        if start_string_id >= 0 and stop_string_id >= 0:
            break
    
    output_string = ""
    str_id = start_string_id
    while str_id <= stop_string_id:
        if str_id == start_string_id:
            output_string += array[str_id][start_id_in_string:]
            continue
        if str_id == stop_id_in_string:
            output_string += array[str_id][stop_id_in_string:]
            continue
        output_string += array[str_id]
    return output_string


def chat_with_model(model, random_seed=42):
    text_collector_callback = PrintTokensCallback(device)
    enc = tiktoken.get_encoding("gpt2")
    
    context_size = 1024

    dialogue = ""
    while True:
        user_text = input("You: ")
        dialogue += "A: " + user_text + "B:"
                
        if len(dialogue) > context_size:
            dialogue = dialogue[:-context_size:]
        
        print("AI: ", end="")

        tokens = enc.encode(dialogue)
        tokens = torch.tensor(tokens).to(device).unsqueeze(0)
        model.generate_seq(tokens, device, text_collector_callback, random_seed=random_seed)
        print("")

        dialogue += text_collector_callback.get_text()


def load_model(model_dir: str):
    os.makedirs("models", exist_ok=True)

    model_path = os.path.join(model_dir, "convgpt2/dialog_gpt2/last.pt")
    if not os.path.exists(os.path.join(model_dir, "convgpt2")):
        path = kagglehub.model_download("spectrespect/convgpt2/pyTorch/default")
        shutil.move(path, model_dir)
    return model_path


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = load_model("models")

    model, optimizer_sd, step, epoch = ConvGPT2.from_checkpoint(model_path)
    model.to(device)

    text_collector_callback = PrintTokensCallback(device)

    chat_with_model(model, 42)
    