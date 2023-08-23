import torch
import torch.nn as nn
import numpy as np
import sys

from model.attention import MultiHeadAttention
from model.common import PositionalEncoding, LayerNorm, PositionwiseFFN

class Transformer(nn.Module):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
    
    def forward(self, x):
        return x

    
class Encoder(nn.Module):
    def __init__(self, vocab_size=37000, num_layers=6, max_len=256, embed_dim=512, ffn_dim=2048, num_heads=8, eps=1e-5, drop_prob=0.1, device='cpu'):
        super(Encoder, self).__init__()

        self.pos_encoding = PositionalEncoding(max_len, embed_dim, device)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, ffn_dim, num_heads, eps, drop_prob) for _ in range(num_layers)])
    
    def forward(self, inputs, attn_mask=None):
        inputs = self.embedding(inputs)

        # (b, seq_len, embed_dim)
        enc_inputs = inputs + self.pos_encoding(inputs) # input + positional encoding

        # (b, seq_len, embed_dim)
        out = self.layers(enc_inputs, attn_mask) # feed to encoder layers
        return out
    
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=512, ffn_dim=2048, num_heads=8, eps=1e-5, drop_prob=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads) 
        self.norm1 = LayerNorm(embed_dim, eps)
        self.ffn = PositionwiseFFN(embed_dim, ffn_dim, drop_prob)
        self.norm2 = LayerNorm(embed_dim, eps)
    
    def forward(self, inputs, attn_mask=None):
        attn_out, attn_prob = self.attention(inputs, inputs, inputs, attn_mask) # multi-head attention
        attn_out = self.norm1(inputs + attn_out) # add + layernorm
        
        ffn_out = self.ffn(attn_out) # feed forward network
        ffn_out = self.norm2(attn_out + ffn_out) # add + layernorm

        return ffn_out, attn_prob


class Decoder(nn.Module):
    def __init__(self, vocab_size=37000, num_layers=6, max_len=256, ffn_dim=512, hidden_dim=2048, num_heads=8, eps=1e-5, drop_prob=0.1, device='cpu'):
        super(Decoder, self).__init__()

        self.pos_encoding = PositionalEncoding(max_len, ffn_dim, device)
        self.embedding = nn.Embedding(vocab_size, ffn_dim)
        self.layers = nn.ModuleList([EncoderLayer(ffn_dim, hidden_dim, num_heads, eps, drop_prob) for _ in range(num_layers)])
    
    def forward(self, dec_inputs, enc_inputs, self_attn_mask=None, cross_attn_mask=None):
        inputs = self.embedding(inputs)

        # (b, seq_len, embed_dim)
        enc_inputs = inputs + self.pos_encoding(inputs) # input + positional encoding

        # (b, seq_len, embed_dim)
        out = self.layers(dec_inputs, enc_inputs, self_attn_mask, cross_attn_mask) # feed to encoder layers
        return out
    

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, ffn_dim=2048, num_heads=8, eps=1e-5, drop_prob=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(embed_dim, num_heads) 
        self.norm1 = LayerNorm(embed_dim, eps)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads) 
        self.norm2 = LayerNorm(embed_dim, eps)
        self.ffn = PositionwiseFFN(embed_dim, ffn_dim, drop_prob)
        self.norm = LayerNorm(embed_dim, eps)
    
    def forward(self, dec_inputs, enc_inputs, self_attn_mask=None, cross_attn_mask=None):
        self_attn_out, self_attn_prob = self.self_attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask) # multi-head attention
        self_attn_out = self.norm1(dec_inputs + self_attn_out) # add + layernorm

        cross_attn_out, cross_attn_prob = self.cross_attention(self_attn_out, enc_inputs, enc_inputs, cross_attn_mask) # multi-head attention
        cross_attn_out = self.norm2(self_attn_out + cross_attn_out) # add + layernorm
        
        ffn_out = self.ffn(cross_attn_out) # feed forward network
        ffn_out = self.norm3(cross_attn_out + ffn_out) # add + layernorm

        return ffn_out, self_attn_prob, cross_attn_prob
    