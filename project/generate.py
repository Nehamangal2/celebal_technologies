import os
import argparse
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig

def main():
    parser = argparse.ArgumentParser(description="Generate text from trained mini GPT-2 model checkpoint.")
    parser.add_argument("--prompt", type=str, default="ROMEO:", help="The prompt to seed the generation.")
    parser.add_argument("--num_tokens", type=int, default=300, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (1.0 = normal, < 1.0 = more deterministic, > 1.0 = more random).")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling threshold (filter out unlikely tokens).")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt", help="Path to the model checkpoint.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for generation: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found. Please train the model first by running train.py.")
        return

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Recreate the model config and model architecture
    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    model = GPT(config)
    
    # Load model state dict
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # Recreate tokenizer mapping
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join([itos[i] for i in l])

    print(f"Prompt: {repr(args.prompt)}")
    print(f"Generating {args.num_tokens} tokens with temperature={args.temperature}, top_k={args.top_k}...")

    # Encode the prompt
    encoded_prompt = encode(args.prompt)
    if len(encoded_prompt) == 0:
        # Fallback if empty prompt or all chars are out of vocab
        encoded_prompt = [stoi.get("\n", 0)]
    
    idx = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0) # (1, T)

    # Generate tokens autoregressively
    with torch.no_grad():
        for _ in range(args.num_tokens):
            # Crop index context if it exceeds the maximum context length (block_size)
            idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            
            # Get logits at the final time step
            logits = logits[:, -1, :] # (1, vocab_size)
            
            # Apply temperature scaling
            logits = logits / args.temperature
            
            # Apply top-k filtering
            if args.top_k is not None and args.top_k > 0:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                # Mask values below the top-k threshold
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (1, vocab_size)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1) # (1, 1)
            
            # Append next token to sequence
            idx = torch.cat((idx, idx_next), dim=1)

    # Decode and print generated text
    generated_text = decode(idx[0].tolist())
    print("\n" + "="*40 + " GENERATED TEXT " + "="*40)
    print(generated_text)
    print("="*96 + "\n")

if __name__ == "__main__":
    main()
