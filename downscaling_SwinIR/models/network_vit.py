
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT


__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-20"

"""
The implementation is based on the tutorial from the following link
https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
"""

import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):

    def __init__(self,
                 in_channels: int = 8,
                 patch_size: int = 4,
                 emb_size :int = 768,
                 enable_cnn: bool = False,
                 img_size: int = 16):
        super.__init__()
        """
        in_channels :  the number of variables/channles
        patch_size  :  the size of each patch
        emb_size    :  the embedding size
        enable_cnn  :  if use convolutional network as projection, if false, using linear projection
        """
        self.patch_size = patch_size
        if enable_cnn:
            self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
            )

        else:
            self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
            )
            # I remove the cls enbedding, since wwe do not have class labels in the data,
            # I used learnable position embedding in this case
            # In the future, we can include the datetime and location as embedding, it should be implemented here
            self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) ->Tensor:
        x = self.projection(x)
        print("The shape after Patching Embedding",x.shape)
        # add position embedding
        x += self.position
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self,
                 emb_size: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0):
        super().__init__()
        """
        Note, the emb_size is different from the embed_size in PatchEmbedding, the size of  this embedding is the same 
        for the query, values and keys
        """
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)


    def forward(self,
                x:Tensor,
                mask:Tensor)->Tensor:

        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h = self.num_heads, qkv = 3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim = -1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int,
                 expansion: int = 4,
                 drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])




class UpsampleOneStep(nn.Sequential):
    """
    UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
    Used in lightweight SR to save parameters.
    scale (int): Scale factor. Supported scales: 2^n and 3.
    num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops





        