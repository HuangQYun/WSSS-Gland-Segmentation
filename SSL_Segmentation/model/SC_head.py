import torch

import torch.nn as nn
from einops import rearrange

from model.Attention import AdditiveAttention


class MulTaskHead(nn.Module):
    def __init__(self, embedding_size=1536, seg_class=2, num_classes=5):
        super(MulTaskHead, self).__init__()
        head_pipeline = []
        current_embedding_size = embedding_size
        for i in range(3):
            head_pipeline.append(nn.Upsample(scale_factor=2))
            head_pipeline.append(nn.Dropout(0.25))
            head_pipeline.append(nn.Conv2d(current_embedding_size, current_embedding_size // 2, (3, 3), padding=(1, 1)))
            current_embedding_size //= 2
        head_pipeline.append(nn.Upsample(scale_factor=2))
        head_pipeline.append(nn.Dropout(0.25))
        head_pipeline.append(nn.Conv2d(current_embedding_size, seg_class, (3, 3), padding=(1, 1)))
        self.seg_head = nn.Sequential(*head_pipeline)
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(112*112, 1536, bias=False),
            nn.Tanh()
        )
        self.cls_head = nn.Linear(1536, num_classes)
        self.attention = AdditiveAttention(1536, 1536, 384)


    def forward(self, x):  # [12, 1, 197, 1536]
        # b, c, p, e = x.size()
        # cls = x[:, 0, 0, :]
        # patch_norm_token = x[:, 0, 1:, :]  # [12, 196, 1536]
        #
        # bep = patch_norm_token.permute(0, 2, 1)
        # behw = rearrange(bep, 'b l (h w) -> b l h w', h=14, w=14)
        #
        # seg_out = self.seg_head(behw)
        # seg_out = torch.sigmoid(seg_out)
        # _, mask = torch.max(seg_out, 1)
        # mask = mask.float()
        # pool_feature = self.layer1(mask)
        # pool_feature = pool_feature.reshape(b, -1)
        # pool_feature = self.attention_gate(pool_feature)
        #
        # cls = torch.mul(cls, pool_feature[:, :])
        # cls_out = self.cls_head(cls)
        # return seg_out, cls_out


        b, c, p, e = x.size()
        cls = x[:, 0, 0, :]
        patch_norm_token = x[:, 0, 1:, :]  # [12, 196, 1536]

        bep = patch_norm_token.permute(0, 2, 1)
        behw = rearrange(bep, 'b l (h w) -> b l h w', h=14, w=14)

        seg_out = self.seg_head(behw)
        seg_out = torch.sigmoid(seg_out)

        pool_feature = self.layer1(seg_out)
        pool_feature = pool_feature.reshape(b, 2, -1)

        pool_feature = self.attention_gate(pool_feature)
        # pool_feature = pool_feature[:, 1, :]
        cls = self.attention(pool_feature, cls.unsqueeze(1), cls.unsqueeze(1), 0)
        cls = cls[:, 0, :]
        # cls = torch.mul(cls, pool_feature[:, :])
        cls_out = self.cls_head(cls)
        return seg_out, cls_out


if __name__ == '__main__':
    model = MulTaskHead()
    print(model)
    feats = torch.rand(12, 1, 197, 1536)
    seg_out, cls_out = model(feats)
    print(seg_out.shape, cls_out.shape)