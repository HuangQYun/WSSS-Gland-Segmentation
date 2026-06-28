
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, key, value, choose):
        """
        Args:
            query: (N, n, d_q)
            key: (N, m, d_k)
            value: (N, m, d_v)
        """
        if(choose == 0):
            query, key = self.W_q(query), self.W_k(key)
            # print('qk', query.unsqueeze(2).shape, key.unsqueeze(1).shape)
            features = query.unsqueeze(2) + key.unsqueeze(1)
            # print('ff', features.shape)
            features = torch.tanh(features)
            scores = self.W_v(features).squeeze(-1)
            # print(scores.shape)
            attn_weights = F.softmax(scores, dim=1)
            # print('value', value.shape, attn_weights.shape)
        else:
            attn_weights = F.softmax(torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.size(2)), dim=-1)
        return torch.bmm(attn_weights, value)


if __name__ == '__main__':
    q = torch.rand(12, 2, 1536)
    k = torch.rand(12, 1, 1536)
    v = torch.rand(12, 1, 1536)
    model = AdditiveAttention(1536, 1536, 384)
    out = model(q, k, v, 0)
    print(out.shape)
