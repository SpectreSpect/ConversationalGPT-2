from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import tiktoken
from transformers import GPT2LMHeadModel


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TokenGenerationCallback():
    
    def __init__(self):
        pass

    def __call__(self, token) -> bool:
        pass


class TextCollectorCallback(TokenGenerationCallback):
    
    def __init__(self, device, generation_length: int = -1, termination_sequences: list = ["A:", "B:", "<|endoftext|>"]):
        self.generation_length = generation_length
        self.termination_sequences = termination_sequences
        self.termination_token_sequences = []

        self.enc = tiktoken.get_encoding("gpt2")
        self.generated_text = ""
        self.generated_idx = torch.empty((1, 0), device=device)
        self.num_generated_tokens = 0
        
    def __call__(self, token) -> bool:
        self.generated_idx = torch.cat((self.generated_idx, token), dim=1) 

        decoded_token = self.enc.decode(token[0, :].tolist())
        self.generated_text += decoded_token

        self.num_generated_tokens += 1
        
        if self.generation_length >= 0 and self.num_generated_tokens >= self.generation_length:
            return True

        for seq in self.termination_sequences:
            if self.generated_text[-len(seq):] == seq:
                self.generated_text = self.generated_text[:-len(seq)]
                return True

        return False

    def get_text(self):
        return self.generated_text
    
    def get_idx(self):
        return self.generated_idx


class PrintTokensCallback(TokenGenerationCallback):
    
    def __init__(self, device, generation_length: int = -1, termination_sequences: list = ["A:", "B:", "<|endoftext|>"]):
        self.generation_length = generation_length
        self.termination_sequences = termination_sequences
        self.termination_token_sequences = []

        self.enc = tiktoken.get_encoding("gpt2")
        self.generated_tokens = []
        self.generated_text = ""
        self.generated_idx = torch.empty((1, 0), device=device)
        self.num_generated_tokens = 0
        self.num_generated_chars = 0
        self.token_to_print_id = 0
        self.num_printed_chars = 0

        self.longest_term_sequence_length = 0
        for seq in termination_sequences:
            if len(seq) > self.longest_term_sequence_length:
                self.longest_term_sequence_length = len(seq)
        
    def __call__(self, token) -> bool:
        self.generated_idx = torch.cat((self.generated_idx, token), dim=1) 

        decoded_token = self.enc.decode(token[0, :].tolist())
        self.generated_tokens.append(decoded_token)
        self.generated_text += decoded_token

        self.num_generated_tokens += 1
        self.num_generated_chars += len(decoded_token)

        terminate = False


        # length = 0
        # for i in range(self.generated_tokens):
        #     length += self.generated_tokens[-i]
        #     if length >= self.longest_term_sequence_length:


        
        num_chars_to_print = self.num_generated_chars - self.longest_term_sequence_length

        if self.generation_length >= 0 and self.num_generated_tokens >= self.generation_length:
            num_chars_to_print = self.num_generated_chars
            terminate = True

        for seq in self.termination_sequences:
            if self.generated_text[-len(seq):] == seq:
                self.generated_text = self.generated_text[:-len(seq)]
                self.generated_tokens[-1] = self.generated_tokens[-1][:-len(seq)]
                num_chars_to_print = self.num_generated_chars - len(seq)
                terminate = True
                break
        
        if self.num_generated_chars > self.longest_term_sequence_length:
            while self.token_to_print_id < len(self.generated_tokens):
                if self.num_printed_chars + len(self.generated_tokens[self.token_to_print_id]) > num_chars_to_print:
                    break
                
                print(self.generated_tokens[self.token_to_print_id], end="")
                self.num_printed_chars += len(self.generated_tokens[self.token_to_print_id])
                self.token_to_print_id += 1
                
        return terminate

    def get_text(self):
        return self.generated_text
    
    def get_idx(self):
        return self.generated_idx


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class DialogGPT2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size ..."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        loss = None
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = DialogGPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # assert len()
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {}")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
    def generate_token(self, idx: str, device, random_seed=42):
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(random_seed)

        logits, _ = self.forward(idx) # B, T, C
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        
        return xcol
    
    def generate_seq(self, idx, device, callback: TokenGenerationCallback, random_seed=42):
        while True:
            token = self.generate_token(idx, device, random_seed)
            termenation_state = callback(token)
            if termenation_state:
                break
            idx = torch.cat((idx, token), dim=1)


    
    def generate(self, text: str, generation_length: int, termination_sequences, device, stream=False):
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        xgen = torch.tensor(tokens).to(device).unsqueeze(0)

        longest_seq_len = 0
        for seq in termination_sequences:
            longest_seq_len = max(len(seq), longest_seq_len)

        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        decoded = ""
        for i in range(generation_length):
            with torch.no_grad():
                logits, _ = self.forward(xgen) # B, T, C
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)
                decoded_token = enc.decode(xcol[0, :].tolist())
                decoded += decoded_token

                if stream:
                    # if i >= longest_seq_len:
                    print(decoded_token, end="")

                for seq in termination_sequences:
                    if decoded[-len(seq):] == seq:
                        return decoded[:-len(seq)]
        
        # tokens = xgen[0, :].tolist()
        # decoded = enc.decode(tokens)
        return decoded
    
    def generate_callbacked(self, text, callback):
        pass
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
        step = checkpoint['step']
        epoch = checkpoint['epoch']
        optimizer_sd = checkpoint['optimizer']
        return model, optimizer_sd, step, epoch