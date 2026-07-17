import os
import math
import time
import torch
from data import get_batch, vocab_size, stoi, itos
from model import GPT, GPTConfig

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 20
warmup_iters = 200
lr_decay_iters = 5000
learning_rate = 3e-4
min_lr = 3e-5
weight_decay = 0.1
grad_clip = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory for checkpoints
os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = os.path.join("checkpoints", "model.pt")

print(f"Using device: {device}")

# Model initialization
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2
)
model = GPT(config)
model.to(device)

# Configure optimizer
def configure_optimizers(model, weight_decay, learning_rate, betas=(0.9, 0.95)):
    # Start with all parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Decayed parameters: 2D weights (matmul weights, embeddings)
    # Non-decayed parameters: 1D parameters (biases, layernorm weights)
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer

optimizer = configure_optimizers(model, weight_decay, learning_rate)

# Cosine learning rate scheduler with warmup
def get_lr(it):
    # 1) Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) After decay iters, return minimum learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) Cosine decay in between
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Loss estimation utility
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Check if checkpoint exists to resume (and verify compatibility)
start_iter = 0
best_val_loss = float("inf")
if os.path.exists(checkpoint_path):
    try:
        print(f"Checking checkpoint {checkpoint_path} for compatibility...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ckpt_args = checkpoint.get("model_args", {})
        
        # Compare key architecture parameters
        match = (ckpt_args.get("n_layer") == config.n_layer and
                 ckpt_args.get("n_embd") == config.n_embd and
                 ckpt_args.get("n_head") == config.n_head and
                 ckpt_args.get("block_size") == config.block_size)
        
        if match:
            print("Checkpoint matches architecture. Resuming training...")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_iter = checkpoint["iter_num"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            print(f"Resumed from step {start_iter - 1} with best val loss: {best_val_loss:.4f}")
        else:
            print("Checkpoint architecture mismatch. Deleting incompatible checkpoint and starting from scratch...")
            os.remove(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint, starting from scratch: {e}")

# Training Loop
t0 = time.time()

print("Starting training...")
for iter_num in range(start_iter, max_iters + 1):
    # Determine learning rate for this iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluate train/val loss
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")
        
        # Checkpoint if we have a new best validation loss
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            if iter_num > start_iter or start_iter == 0: # Avoid saving initial checkpoint on resumed run unless better
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": {
                        "vocab_size": config.vocab_size,
                        "block_size": config.block_size,
                        "n_layer": config.n_layer,
                        "n_head": config.n_head,
                        "n_embd": config.n_embd,
                        "dropout": config.dropout
                    },
                    "stoi": stoi,
                    "itos": itos,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss
                }
                print(f"Saving checkpoint to {checkpoint_path} (val loss: {best_val_loss:.4f})")
                torch.save(checkpoint, checkpoint_path)

    # Fetch a batch of training data
    x, y = get_batch("train", batch_size, block_size, device)

    # Forward pass
    logits, loss = model(x, y)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    optimizer.step()

t1 = time.time()
print(f"Training completed in {(t1 - t0)/60:.2f} minutes.")
print(f"Best validation loss achieved: {best_val_loss:.4f}")
