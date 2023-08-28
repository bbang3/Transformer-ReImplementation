import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        batch_size, num_heads, seq_len, d_k = K.shape # num_heads * d_k = 모든 head를 합친 attention 크기

        # inner product of query and key + scaling
        # (bs, num_heads, seq_q, d_k) * (bs, num_heads, d_k, seq_k) => (bs, num_heads, seq_q, seq_k)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (d_k ** 0.5) 
    
        # mask: boolean matrix
        if mask is not None:
            attn_scores.masked_fill_(mask, -float('inf')) # fill -inf to masked positions

        attn_prob = self.softmax(attn_scores) # take softmax to get attention distribution
        V = torch.matmul(attn_prob, V) # weighted sum of value

        return V, attn_prob
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.scaled_dot_atttn = ScaledDotProductAttention()

        # embed_dim = num_heads * d_head
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V, mask=None):
        # (batch_size, seq_len, embed_dim = num_heads * d_head)
        bs = Q.shape[0]
        Q, K, V = self.W_q(Q), self.W_k(K), self.W_v(V)

        # (batch_size, num_heads, seq_len, d_head)
        Q = Q.view(bs, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(bs, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(bs, -1, self.num_heads, self.d_head).transpose(1, 2)

        # (bs, seq_len, seq_len) -> (bs, num_heads, seq_len, seq_len)
        mask = mask.unsqueeze(1).expand(bs, self.num_heads, -1, -1)

        # out: (batch_size, num_heads, seq_len_q, d_head)
        # attn_prob: (bs, num_heads, seq_len_q, seq_len_k)
        out, attn_prob = self.scaled_dot_atttn(Q, K, V, mask=mask)
        # (batch_size, seq_len, embed_dim)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_head) # To use view(), contiguous() is needed
        out = self.W_o(out)

        # (batch_size, seq_len, embed_dim)
        return out, attn_prob