import torch
import torch.nn as nn
import numpy as np

class Transformer(nn.Module):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
    
    def forward(self, x):
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, eps=1e-5, drop_prob=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim, eps)
        self.ffn = PositionwiseFFN(embed_dim, embed_dim * 4, drop_prob)
        self.norm2 = LayerNorm(embed_dim, eps)
    
    def forward(self, inputs, attn_mask=None):
        attn_out, attn_prob = self.attention(inputs, inputs, inputs, attn_mask)
        attn_out = self.norm1(inputs + attn_out)
        
        ffn_out = self.ffn(attn_out)
        ffn_out = self.norm2(attn_out + ffn_out)

        return ffn_out, attn_prob

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim, device) -> None:
        super(PositionalEncoding, self).__init__()

        self.encoding = self._get_encoding(max_len, embed_dim)

    def forward(self, x: torch.tensor):
        batch_size, seq_len = x.shape

        return self.encoding[:seq_len, :]

    def _get_encoding(self, max_len, embed_dim, device):
        encoding = torch.zeros(max_len, embed_dim, device=device)
        encoding.requires_grad = False # freeze

        pos = torch.arange(0, max_len, device=device).float()
        pos = pos.unsqueeze(dim=1) # (seq_len, 1)

        _2i = torch.arange(0, embed_dim, step=2).float() # (embed_dim//2,)

        # broadcast: (seq_len, 1) / (1, embed_dim//2) => (seq_len, embed_dim//2)
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_dim))) 
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_dim)))

        return encoding
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, Q, K, V, mask=None):
        batch_size, num_heads, seq_len, d_k = K.shape # num_heads * d_k = entire K shape

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (d_k ** 0.5)

        if mask is not None:
            attn_scores.masked_fill_(mask, -float('inf')) # mask: boolean matrix

        attn_prob = self.softmax(attn_scores)
        V = torch.matmul(attn_prob, V)

        return V, attn_prob
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.scaled_dot_atttn = ScaledDotProductAttention()

        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V, mask=None):
        # (batch_size, seq_len, embed_dim, num_heads * d_head)
        bs = Q.shape[0]
        Q, K, V = self.W_q(Q), self.W_k(K), self.W_v(V)

        # (batch_size, num_heads, seq_len, d_head)
        Q = Q.view(bs, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(bs, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(bs, -1, self.num_heads, self.d_head).transpose(1, 2)

        # out: (batch_size, num_heads, seq_len, d_head)
        # attn_prob: (bs, num_heads, seq_len_q, seq_len_k)
        out, attn_prob = self.scaled_dot_atttn(Q, K, V, mask=mask)
        # (batch_size, seq_len, embed_dim)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_head) # To use view(), contiguous() is needed
        out = self.W_o(out)

        # (batch_size, seq_len, embed_dim)
        return out, attn_prob
    
class LayerNorm(nn.Module):
    def __init__(self, embed_dim=512, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta

        return out


class PositionwiseFFN(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=2048, drop_prob=0.1): # hidden_dim = 4 * embed_dim = 2048
        super(PositionwiseFFN, self).__init__()

        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=drop_prob)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out