import torch

import torch.nn as nn


class SSLCLSHead(nn.Module):
    def __init__(self, n_class=2):
        super(SSLCLSHead, self).__init__()
        self.head = nn.Linear(1536, n_class)

    def forward(self, x):
        output = self.head(x)
        return output


if __name__ == '__main__':
    model = SSLCLSHead()

    feats = torch.rand(1, 1536, 14, 14)
    out = model(feats)
    print(out.shape)