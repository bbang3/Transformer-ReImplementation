import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.common import PositionalEncoding, LayerNorm, PositionwiseFFN

class Transformer(nn.Module):
    def __init__(self, 
                enc_vocab_size=37000, 
                dec_vocab_size=37000,
                num_layers=6, 
                max_len=256, 
                embed_dim=512, 
                ffn_dim=2048, 
                num_heads=8, 
                eps=1e-5, 
                drop_prob=0.1, 
                device='cpu'
                ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(enc_vocab_size, num_layers, max_len, embed_dim, ffn_dim, num_heads, eps, drop_prob, device)
        self.decoder = Decoder(dec_vocab_size, num_layers, max_len, embed_dim, ffn_dim, num_heads, eps, drop_prob, device)
        self.linear = nn.Linear(embed_dim, dec_vocab_size)

        self.device = device
    
    def forward(self, enc_inputs, dec_inputs):
        enc_mask = self.get_padding_mask(enc_inputs, enc_inputs) # for self-attention in encoder

        dec_enc_mask = self.get_padding_mask(dec_inputs, enc_inputs) # for cross-attention in decoder
        dec_mask = self.get_padding_mask(dec_inputs, dec_inputs).bool() | self.get_lookhead_mask(dec_inputs).bool() # for self-attention in decoder

        enc_outputs, enc_attn_prob  = self.encoder(enc_inputs, attn_mask=enc_mask)
        dec_outputs, dec_attn_prob, cross_attn_prob = self.decoder(dec_inputs, enc_outputs, self_attn_mask=dec_mask, cross_attn_mask=dec_enc_mask)

        outputs = self.linear(dec_outputs) # (bs, seq_len, embed_dim) -> (bs, seq_len, dec_vocab_size)
        return outputs
    
    def get_padding_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.shape
        batch_size, len_k = seq_k.shape
        attn_mask = seq_k.eq(0) # (bs, len_k)
        
        # (bs, len_q, len_k)
        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
        return attn_mask
    
    def get_lookhead_mask(self, seq: torch.Tensor):
        batch_size, seq_len = seq.shape

        lookahead_mask = torch.ones((batch_size, seq_len, seq_len), device=self.device).bool() # (b, seq_len, seq_len)
        lookahead_mask = lookahead_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D) including diagonal

        return lookahead_mask


class Encoder(nn.Module):
    def __init__(self, vocab_size=37000, num_layers=6, max_len=256, embed_dim=512, ffn_dim=2048, num_heads=8, eps=1e-5, drop_prob=0.1, device='cpu'):
        super(Encoder, self).__init__()

        self.pos_encoding = PositionalEncoding(max_len, embed_dim, device)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, ffn_dim, num_heads, eps, drop_prob) for _ in range(num_layers)])
    
    def forward(self, inputs, attn_mask=None):
        inputs = self.embedding(inputs)

        # (b, seq_len, embed_dim)
        outputs = inputs + self.pos_encoding(inputs) # input + positional encoding

        # (b, seq_len, embed_dim)
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask) # feed to encoder layers

        return outputs, attn_prob
    
    
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
    def __init__(self, vocab_size=37000, num_layers=6, max_len=256, embed_dim=512, ffn_dim=2048, num_heads=8, eps=1e-5, drop_prob=0.1, device='cpu'):
        super(Decoder, self).__init__()

        self.pos_encoding = PositionalEncoding(max_len, embed_dim, device)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, ffn_dim, num_heads, eps, drop_prob) for _ in range(num_layers)])
    
    def forward(self, dec_inputs, enc_outputs, self_attn_mask=None, cross_attn_mask=None):
        dec_inputs = self.embedding(dec_inputs)

        # (b, seq_len, embed_dim)
        dec_outputs = dec_inputs + self.pos_encoding(dec_inputs) # input + positional encoding

        # (b, seq_len, embed_dim)
        for layer in self.layers:
            dec_outputs, self_attn_prob, cross_attn_prob = layer(dec_outputs, enc_outputs, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask) # feed to decoder layers
        return dec_outputs, self_attn_prob, cross_attn_prob
    

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, ffn_dim=2048, num_heads=8, eps=1e-5, drop_prob=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(embed_dim, num_heads) 
        self.norm1 = LayerNorm(embed_dim, eps)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads) 
        self.norm2 = LayerNorm(embed_dim, eps)
        self.ffn = PositionwiseFFN(embed_dim, ffn_dim, drop_prob)
        self.norm3 = LayerNorm(embed_dim, eps)
    
    def forward(self, dec_inputs, enc_inputs, self_attn_mask=None, cross_attn_mask=None):
        self_attn_out, self_attn_prob = self.self_attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask) # multi-head attention
        self_attn_out = self.norm1(dec_inputs + self_attn_out) # add + layernorm

        cross_attn_out, cross_attn_prob = self.cross_attention(self_attn_out, enc_inputs, enc_inputs, cross_attn_mask) # multi-head attention
        cross_attn_out = self.norm2(self_attn_out + cross_attn_out) # add + layernorm
        
        ffn_out = self.ffn(cross_attn_out) # feed forward network
        ffn_out = self.norm3(cross_attn_out + ffn_out) # add + layernorm

        return ffn_out, self_attn_prob, cross_attn_prob