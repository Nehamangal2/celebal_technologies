import os
import urllib.request
import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"

def download_data():
    if not os.path.exists(DATA_FILE):
        print(f"Downloading Tiny Shakespeare dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")
    else:
        print("Tiny Shakespeare dataset already exists.")

# Ensure data is downloaded
download_data()

# Read the dataset to construct tokenizer
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mappings from characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(l: list[int]) -> str:
    return ''.join([itos[i] for i in l])

# Convert text to PyTorch tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split (90% train, 10% val)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str, batch_size: int, block_size: int, device: torch.device):
    """
    Generate a small batch of data of inputs x and targets y.
    """
    split_data = train_data if split == 'train' else val_data
    # Sample random starting indices
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    # Vectorized indexing
    idx = ix.unsqueeze(1) + torch.arange(block_size).unsqueeze(0)
    x = split_data[idx]
    y = split_data[idx + 1]
    return x.to(device), y.to(device)

if __name__ == "__main__":
    print(f"Dataset size: {len(text)} characters")
    print(f"Vocab size: {vocab_size}")
    print(f"Unique characters: {''.join(chars)}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    
    # Test batch generation
    x_test, y_test = get_batch('train', 4, 8, torch.device('cpu'))
    print("Test Batch x shape:", x_test.shape)
    print("Test Batch y shape:", y_test.shape)
    print("x[0]:", x_test[0].tolist())
    print("y[0]:", y_test[0].tolist())
    print("Decoded x[0]:", repr(decode(x_test[0].tolist())))
    print("Decoded y[0]:", repr(decode(y_test[0].tolist())))
