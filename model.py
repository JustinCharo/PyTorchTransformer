import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Creates built in embedding layer that converts input tokens into vectors of size 512
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # Dropout layer to prevent overfitting

        # Create a 2D tensor or matrix filled with zeros of shape (seq_len, d_model)
        PE = torch.zeros(seq_len, d_model)
        # Create a 1D tensor or vector of shape (seq_len, 1) that contains the position indices
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # tensor of shape (d_model/2) to scale the position indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # Apply sin to even positions
        PE[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd positions
        PE[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension to PE
        PE = PE.unsqueeze(0) # tensor of shape (1, seq_len, d_model)

        self.register_buffer('PE', PE) # saves the state of the model but not a parameter

    # Adds the postion encodings to the input embeddings, and then applies dropout
    def forward(self, x):
        x = x + (self.PE[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

