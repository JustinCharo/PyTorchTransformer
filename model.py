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

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6):
        suoer().__init__()
        # small value to avoid division by zero
        self.eps = eps
        # multiplier to scale the normalized output
        self.alpha = nn.Parameter(torch.ones(1))
        # added bias to the normalized output
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout: float):
        super().__init__()
        self.droupout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # tensor of shape (batch_size, Seq_len, d_model)
        # maps the input tensor through linear_1 to another tensor of higher dimension of shape (batch_size, seq_len, d_ff)
        # apply linear 2 which will map the tensor back to the original dimension of shape (batch_size, seq_len, d_model)
        # apply dropout to the output of linear_2
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.droupout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).tranpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).tranpose(1, 2)

        x, self.attention_scores = self.MultiHeadAttentionBlock(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h + self.d_k)

        return self.w_o(x)

