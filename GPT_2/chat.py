import os
import sys
import argparse
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig

def main():
    parser = argparse.ArgumentParser(description="Interactive Mini GPT-2 Text Generator.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering threshold (default: 40)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt", help="Path to checkpoint file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found. Please train the model first.")
        return

    print(f"Loading model checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join([itos[i] for i in l])

    print("\n" + "="*50)
    print("            MINI GPT-2 TEXT GENERATOR            ")
    print("="*50)
    print(f"Configuration: temperature={args.temperature}, top_k={args.top_k}")
    print("Type any starting text, and the model will generate a continuation.")
    print("Type '/reset' to clear conversation history, or 'exit' to quit.\n")

    conversation_history = ""

    while True:
        try:
            user_input = input("\nPrompt> ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if user_input.strip().lower() == "exit":
            print("Exiting. Goodbye!")
            break

        if user_input.strip().lower() == "/reset":
            conversation_history = ""
            print("--- Conversation history cleared! ---\n")
            continue

        if not user_input:
            continue

        # Append user prompt to rolling conversation history
        conversation_history += user_input

        # Limit history to model's context length (block_size characters)
        history_chars = conversation_history[-config.block_size:]
        idx = torch.tensor(encode(history_chars), dtype=torch.long, device=device).unsqueeze(0)

        print("Completion: ", end="", flush=True)
        
        generated_reply = ""
        max_reply_tokens = 200

        with torch.no_grad():
            for _ in range(max_reply_tokens):
                # Crop context if it exceeds block size
                idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
                
                # Forward pass to get logits of the last token
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / args.temperature
                
                # Apply top-k filtering
                if args.top_k is not None and args.top_k > 0:
                    v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                
                # Sample next character
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                idx = torch.cat((idx, idx_next), dim=1)
                
                # Stream character output
                char = decode(idx_next[0].tolist())
                sys.stdout.write(char)
                sys.stdout.flush()
                
                generated_reply += char

        print()
        # Save generated continuation to rolling history
        conversation_history += generated_reply

if __name__ == "__main__":
    main()
