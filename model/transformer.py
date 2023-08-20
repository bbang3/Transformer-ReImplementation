import torch
import torch.nn as nn
import numpy as np

from model.attention import MultiHeadAttention
from model.common import PositionalEncoding, LayerNorm, PositionwiseFFN

class Transformer(nn.Module):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
    
    def forward(self, x):
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size=37000, num_layers=6, max_len=256, embed_dim=512, hidden_dim=2048, num_heads=8, eps=1e-5, drop_prob=0.1, device='cpu'):
        super(Encoder, self).__init__()

        self.pos_encoding = PositionalEncoding(max_len, embed_dim, device)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, eps, drop_prob) for _ in range(6)])
    
    def forward(self, inputs, attn_mask=None):
        inputs = self.embedding(inputs)
        enc_inputs = inputs + self.pos_encoding(inputs)

        out = self.layers(enc_inputs, attn_mask)
        return out
    
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=2048, num_heads=8, eps=1e-5, drop_prob=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim, eps)
        self.ffn = PositionwiseFFN(embed_dim, hidden_dim, drop_prob)
        self.norm2 = LayerNorm(embed_dim, eps)
    
    def forward(self, inputs, attn_mask=None):
        attn_out, attn_prob = self.attention(inputs, inputs, inputs, attn_mask)
        attn_out = self.norm1(inputs + attn_out)
        
        ffn_out = self.ffn(attn_out)
        ffn_out = self.norm2(attn_out + ffn_out)

        return ffn_out, attn_prob


