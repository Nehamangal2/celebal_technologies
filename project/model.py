import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    def __init__(self, vocab_size=65, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = 256
        self.n_layer = 6
        self.n_head = 6
        self.n_embd = 384
        self.dropout = 0.2
        for k, v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        # Regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Causal mask: lower triangular matrix of ones
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch and split them
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to (B, T, n_head, head_size) and transpose to (B, n_head, T, head_size)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_drop(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LayerNorm block structure with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing: tie weights of lm_head to token embedding (wte)
        self.lm_head.weight = self.transformer.wte.weight

        # Init all weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Initialized GPT model. Total Parameters: {self.get_num_params()/1e6:.4f}M")

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For weight-tied models, PyTorch's self.parameters() only returns the shared wte
        tensor once, which gives the true parameter count.
        """
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        
        # Position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)

        # Token and position embeddings
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # (1, t, n_embd)
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        # Final LayerNorm
        x = self.transformer.ln_f(x)
        
        # Language model head to get logits
        logits = self.lm_head(x) # (b, t, vocab_size)

        # Loss calculation if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

if __name__ == "__main__":
    # Test model initialization and forward pass
    config = GPTConfig(vocab_size=65)
    model = GPT(config)
    
    # Dummy input (batch_size=4, sequence_length=32)
    dummy_input = torch.randint(0, 65, (4, 32))
    dummy_targets = torch.randint(0, 65, (4, 32))
    
    logits, loss = model(dummy_input, dummy_targets)
    print("Logits shape:", logits.shape)
    print("Loss value:", loss.item())
    
    # Check weight sharing
    is_shared = (model.lm_head.weight is model.transformer.wte.weight)
    print("LM Head and Token Embedding weights shared:", is_shared)
