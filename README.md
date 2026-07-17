# Mini GPT-2 Decoder-Only Transformer from Scratch

This project implements a mini GPT-2 style decoder-only transformer language model from scratch in PyTorch, inspired by Andrej Karpathy's `nanoGPT`. It is trained on the character-level Tiny Shakespeare dataset.

---

## Architectural Details

The transformer architecture is built manually in PyTorch without using any high-level pretrained models or HuggingFace transformers libraries.

1. **Token and Positional Embeddings**:
   * Token embedding lookup maps character indices to a continuous space of dimension `n_embd` (`128`).
   * A learned positional embedding layer maps positions `0` to `block_size - 1` (`128`) to the same dimension.
   * Dropout is applied to the sum of token and positional embeddings.

2. **Manual Multi-Head Causal Self-Attention**:
   * Computes Query, Key, and Value projections using a single linear layer.
   * Splits projections into multiple attention heads (`n_head = 4`).
   * Computes scaled dot-product attention manually: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$.
   * A lower triangular mask matrix (`tril`) is registered to mask future tokens (`masked_fill(tril == 0, -inf)`), forcing the attention to be causal.
   * Attention weights and residual pathways include dropout for regularization.

3. **Transformer Block**:
   * Implements a **Pre-LayerNorm** architecture, which is more stable than post-LayerNorm:
     * $x_{mid} = x + \text{Attention}(\text{LayerNorm}(x))$
     * $x_{out} = x_{mid} + \text{MLP}(\text{LayerNorm}(x_{mid}))$
   * The Feed-Forward Network (MLP) consists of an expansion linear layer (to $4 \times d_{model}$), a GELU activation, a projection back to $d_{model}$, and dropout.

4. **Main Stack & Weight Tying**:
   * Sequential stack of $N$ (`4`) transformer blocks.
   * Final LayerNorm before projecting to vocabulary logits.
   * **Weight Sharing (Tying)**: The weights of the input token embedding layer and the final language model linear head are tied/shared directly (`lm_head.weight = wte.weight`), reducing model footprint and improving convergence.
   * **Weight Initialization**: Initialized linear and embedding weights with a normal distribution ($\sigma = 0.02$). Scaled output projection weights of residual blocks by $1/\sqrt{2 \times n_{layer}}$ to prevent variance explosion during initialization.

---

## Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| `n_layer` | 4 | Number of transformer blocks |
| `n_head` | 4 | Number of attention heads |
| `n_embd` | 128 | Embedding dimensionality |
| `block_size` | 128 | Context/sequence length (tokens) |
| `dropout` | 0.1 | Dropout probability |
| `vocab_size` | 65 | Shakespeare character vocabulary size |
| `batch_size` | 64 | Batch size for training |
| `learning_rate`| 3e-4 | Peak learning rate (with cosine decay) |
| `min_lr` | 3e-5 | Minimum decay learning rate |
| `warmup_iters` | 50 | Linear learning rate warmup iterations |
| `max_iters` | 500 | Total training iterations |
| **Total Params**| **0.8180M** | Lightweight model optimized for CPU training |

---

## File Structure

* **[data.py](file:///c:/Users/neham/OneDrive/Desktop/antigravity_project/data.py)**: Downloads the Tiny Shakespeare dataset, builds character-level `stoi` and `itos` mappings, performs 90/10 train/validation splitting, and handles batch compilation using an optimized vectorized index lookup.
* **[model.py](file:///c:/Users/neham/OneDrive/Desktop/antigravity_project/model.py)**: Implements custom configuration, multi-head causal self-attention, feedforward blocks, weight-tied GPT model, initialization parameters, and testing routines.
* **[train.py](file:///c:/Users/neham/OneDrive/Desktop/antigravity_project/train.py)**: Coordinates the training run, optimizer parameters (weight decay exclusions), cosine learning rate scheduler, model checkpoint saving, and metrics monitoring.
* **[generate.py](file:///c:/Users/neham/OneDrive/Desktop/antigravity_project/generate.py)**: Loads saved checkpoint states, processes seeds, performs top-k sampling with temperature adjustment, and decodes the model's text outputs.

---

## Setup & Running Scripts

### Prerequisites
Install PyTorch (version 2.0+ recommended):
```bash
pip install torch
```

### 1. Download and Prepare Data
Download Shakespeare's text and check character vocab mapping:
```bash
python data.py
```

### 2. Train the Model
Train the model. The best checkpoint will be automatically saved to `checkpoints/model.pt` based on validation loss updates:
```bash
python train.py
```

### 3. Generate Text
Generate text from a prompt. You can adjust temperature (e.g. `0.8` for balance, lower for deterministic) and `top_k` filtering:
```bash
python generate.py --prompt "ROMEO:" --num_tokens 300 --temperature 0.8 --top_k 10
```

---

## Training Run Summary

Training was executed on CPU for **75 steps** (constrained to fit within 25 minutes) on the larger **10.77M parameter model** (batch size **64**, context length **256**, optimized vectorized indexing):
* **Step 0**: Train Loss `4.2999` | Val Loss `4.2958` | lr `0.000000`
* **Step 25**: Train Loss `2.6088` | Val Loss `2.6104` | lr `0.000259`
* **Step 50**: Train Loss `2.5196` | Val Loss `2.5296` | lr `0.000113`
* **Step 75**: Train Loss `2.5020` | Val Loss `2.5038` | lr `0.000030`

**Total Training Time**: ~23.7 minutes of active CPU computation.
**Best Validation Loss**: `2.5038`

### Sample Output (Prompt: `" "`, Temperature: `0.8`, Top-k: `40`)
```text
 doule
F f oromerk, fount, his
Wanof dathiT, t ourefo d s t cofr.
I y ithe hignorshe n be.
A'she s grke y by,
BUSheneakitrer s hthimathe ucor cears thape athacang, fof cades o iforrenon, mme where fincer m, fomyoned whou I ber:
IOFowe cof he ind m ad ce s aicounnthire the wing O ba od t o t all w main thangrnou,
NGoure. thealyondor it are yon sit kes thentouenendinde bre sus pld me the hellsatherootour, od s ICENone co bre ke my hiourse, soullen be the atoouthegivesthenos shele and
Avenof youl t 
```
The model successfully learned the layout conventions, play dialogue formatting, speaker tags (e.g. `BUSheneakitrer:`), and spacing structures, though it requires further training steps to learn complete English word spelling and coherence.
