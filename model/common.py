import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim, device) -> None:
        super(PositionalEncoding, self).__init__()

        self.encoding = self._get_encoding(max_len, embed_dim, device)

    def forward(self, x: torch.tensor):
        batch_size, seq_len, embed_dim = x.shape
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

    
class LayerNorm(nn.Module):
    def __init__(self, embed_dim=512, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        out = (x - mean) / (std + self.eps) # broadcast so that out shape is same as x
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