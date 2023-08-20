import torch
import torch.nn as nn

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