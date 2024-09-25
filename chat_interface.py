from src.models.dialog_gpt2.dialog_gpt2 import DialogGPT2, TextCollectorCallback, PrintTokensCallback
import torch
from dataclasses import dataclass
import tiktoken

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


if __name__ == "__main__":
    device = "cuda"

    # sdfsdf = torch.load("models/dialog_gpt2/last_test.pt", weights_only=False)

    model, optimizer_sd, step, epoch = DialogGPT2.from_checkpoint("models/dialog_gpt2/last_test.pt")
    model.to(device)

    text_collector_callback = PrintTokensCallback(device)

    text = "A: Hello, how are you?B:"
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    idx = torch.tensor(tokens).to(device).unsqueeze(0)


    model.generate_seq(idx, device, text_collector_callback)

    # print(text_collector_callback.get_text())

    # token = model.generate_token(idx, device)

    # text_collector_callback = TextCollectorCallback()
    # text_collector_callback(token)
    # print(text_collector_callback.get_text())

    

    # dialogue = ""
    # while True:
    #     user_text = input("You: ")
    #     dialogue += "A: " + user_text + "B:"
        
    #     if len(dialogue) > 1024:
    #         dialogue = dialogue[:-1024]
        
    #     print("AI: ", end="")
    #     model_response = model.generate(dialogue, 500, ["A:", "B:", "<|endoftext|>"], device, stream=True)
    #     print("\n")

    #     dialogue += model_response
    #     # print("AI: " + model_response)
        