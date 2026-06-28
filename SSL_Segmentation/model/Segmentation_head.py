import torch

import torch.nn as nn
from einops import rearrange


class SSLSegHead(nn.Module):
    def __init__(self, embedding_size=1536, num_classes=2):
        super(SSLSegHead, self).__init__()
        head_pipeline = []
        current_embedding_size = embedding_size
        for i in range(3):
            head_pipeline.append(nn.Upsample(scale_factor=2))
            head_pipeline.append(nn.Dropout(0.25))
            head_pipeline.append(nn.Conv2d(current_embedding_size, current_embedding_size // 2, (3, 3), padding=(1, 1)))
            current_embedding_size //= 2
        head_pipeline.append(nn.Upsample(scale_factor=2))
        head_pipeline.append(nn.Dropout(0.25))
        head_pipeline.append(nn.Conv2d(current_embedding_size, num_classes, (3, 3), padding=(1, 1)))
        self.head = nn.Sequential(*head_pipeline)
        # self.head = nn.Sequential(
        #     nn.Upsample(scale_factor=4),
        #     nn.Dropout(0.25),
        #     nn.Conv2d(embedding_size, 384, (3, 3), padding=(1,1)),
        #     nn.Upsample(scale_factor=4),
        #     nn.Dropout(0.25),
        #     nn.Conv2d(384, num_classes, (3, 3), padding=(1,1)),
        # )

    def forward(self, x):  # [12, 1, 197, 1536]
        patch_norm_token = x[:, 0, 1:, :]  # [12, 196, 1536]
        bep = patch_norm_token.permute(0, 2, 1)
        behw = rearrange(bep, 'b l (h w) -> b l h w', h=14, w=14)
        output = self.head(behw)
        output = torch.sigmoid(output)
        return output


if __name__ == '__main__':
    model = SSLSegHead()
    print(model)
    feats = torch.rand(12, 1, 197, 1536)
    out = model(feats)
    print(out.shape)