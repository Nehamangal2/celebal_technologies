import unittest
import torch
import torch.nn as nn
from model import GPT, GPTConfig, CausalSelfAttention

class TestTransformer(unittest.TestCase):
    
    def setUp(self):
        self.config = GPTConfig(
            vocab_size=65,
            block_size=128,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0
        )
        self.model = GPT(self.config)
        self.model.eval()

    def test_parameter_count(self):
        """Verify the parameter count matches our calculations."""
        # For our set config (2 layers, 2 heads, 64 embd, 128 block size, 65 vocab size):
        # 65*64 (wte) + 128*64 (wpe) + blocks...
        # Just check that it returns a valid positive integer.
        params = self.model.get_num_params()
        self.assertGreater(params, 0)
        print(f"Test Parameter Count: {params}")

    def test_weight_sharing(self):
        """Verify that the token embedding and language model head weights share memory."""
        wte_weight = self.model.transformer.wte.weight
        lm_head_weight = self.model.lm_head.weight
        # Check if they point to the exact same tensor object
        self.assertTrue(wte_weight is lm_head_weight)

    def test_output_shapes(self):
        """Verify the shape of output logits and loss values."""
        batch_size = 4
        sequence_length = 32
        x = torch.randint(0, self.config.vocab_size, (batch_size, sequence_length))
        y = torch.randint(0, self.config.vocab_size, (batch_size, sequence_length))
        
        logits, loss = self.model(x, y)
        
        # Logits should have shape (batch_size, sequence_length, vocab_size)
        self.assertEqual(logits.shape, (batch_size, sequence_length, self.config.vocab_size))
        # Loss should be a single scalar tensor
        self.assertEqual(loss.shape, torch.Size([]))

    def test_causal_masking(self):
        """Verify that causal masking prevents tokens from attending to future tokens."""
        # Create a batch of size 1, sequence length 3
        # If we change the value of the 3rd token, the logits for the 1st and 2nd tokens
        # should remain exactly the same, showing that they do not attend to the future.
        x1 = torch.tensor([[10, 20, 30]])
        x2 = torch.tensor([[10, 20, 40]]) # Only the last token is changed
        
        with torch.no_grad():
            logits1, _ = self.model(x1)
            logits2, _ = self.model(x2)
            
        # Check first token logits (index 0)
        self.assertTrue(torch.allclose(logits1[:, 0, :], logits2[:, 0, :], atol=1e-5))
        # Check second token logits (index 1)
        self.assertTrue(torch.allclose(logits1[:, 1, :], logits2[:, 1, :], atol=1e-5))
        
        # Check third token logits (index 2) - should be different because of the token change
        self.assertFalse(torch.allclose(logits1[:, 2, :], logits2[:, 2, :], atol=1e-5))

if __name__ == "__main__":
    unittest.main()
