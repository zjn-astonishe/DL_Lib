import torch
import math
import torch.nn.functional as F
import torch.nn as nn

from torch import Tensor
from typing import Tuple, Dict

from model.MobileNet import ConvLayer
from model.EdgeViT import LocalAgg, LocalProp, LocalAgg1
# from MobileNet import ConvLayer
# from EdgeViT import LocalAgg, LocalProp, LocalAgg1

class SeparableSelfAttention(nn.Module):
    """
    分离自注意力
    """
    def __init__(self, embed_dim, attn_dropout=0):
        super().__init__()
        self.qkv_proj = ConvLayer(in_channels=embed_dim, out_channels=1+2*embed_dim, kernel_size=1, bias=True)
        # self.qkv_proj = conv_2d(embed_dim, 1+2*embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        # self.qkv_proj = nn.Linear(embed_dim, 1+2*embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = ConvLayer(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=True)
        # self.out_proj = conv_2d(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        # self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)
        context_score = F.softmax(q, dim=-1)
        context_score = self.attn_dropout(context_score)

        context_vector = k * context_score
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(v) * context_vector.expand_as(v)
        out = self.out_proj(out)
        return out

class MyTransformerEncoder(nn.Module):
    """
    MyTransformerEncoder
    """
    def __init__(
        self, 
        embed_dim: int, 
        ffn_latent_dim: int,
        dropout: float = 0,
        ffn_dropout: float = 0,
        attn_drop: float = 0
    ) -> None:
        super().__init__()
        
        # attention
        self.attn = nn.Sequential()
        self.attn.add_module(name="attn_gn", module=nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1))
        self.attn.add_module(name="attn", module=SeparableSelfAttention(embed_dim=embed_dim, attn_dropout=attn_drop))
        self.attn.add_module(name="attn_drop2", module=nn.Dropout(dropout))
        
        # ffn
        self.ffn = nn.Sequential()
        self.ffn.add_module(name="ffn_gn", module=nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1))
        self.ffn.add_module(name="ffn_conv1", module=ConvLayer(in_channels=embed_dim, out_channels=ffn_latent_dim, kernel_size=1, stride=1, bias=True, use_norm=False, use_act=True))
        self.ffn.add_module(name="ffn_drop1", module=nn.Dropout(ffn_dropout))
        self.ffn.add_module(name="ffn_conv2", module=ConvLayer(in_channels=ffn_latent_dim, out_channels=embed_dim, kernel_size=1, stride=1, bias=True, use_norm=False, use_act=False))
        self.ffn.add_module(name="ffn_drop2", module=nn.Dropout(dropout))

    def forward(self, X: Tensor) -> Tensor:
        X = X + self.attn(X)
        X = X + self.ffn(X)
        return X

# class MyTransformerBlock(nn.Module):
#     """
#     中间的block
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         embed_dim: int,
#         ffn_latent_dim: int,
#         dropout: int = 0,
#         attn_dropout: int = 0,
#         patch_h: int = 3,
#         patch_w: int = 3,
#         attn_blocks: int = 3,
#     ) -> None:
        
#         super().__init__()

#         # input rep
#         self.input_rep = nn.Sequential()
#         self.input_rep.add_module(name="input_dwconv", module=ConvLayer(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 stride=1,
#                 kernel_size=3,
#                 groups=in_channels,
#                 use_norm=True,
#                 use_act=True,
#             )
#         )
#         self.input_rep.add_module(name="input_pwconv", module=ConvLayer(
#                 in_channels=in_channels,
#                 out_channels=embed_dim,
#                 kernel_size=1,
#                 stride=1,
#                 use_norm=False,
#                 use_act=False
#             )
#         )

#         # local rep
#         self.local_rep = nn.Sequential()
#         self.local_rep.add_module(name="local_dwconv", module=ConvLayer(
#                 in_channels=embed_dim,
#                 out_channels=embed_dim,
#                 stride=1,
#                 kernel_size=3,
#                 groups=embed_dim,
#                 use_norm=False,
#                 use_act=True,
#             )
#         )

#         # local agg for attn
#         self.local_agg = nn.Sequential()
#         self.local_agg.add_module(name="local_agg", module=LocalAgg1(embed_dim))
        
#         # local prop for attn
#         self.local_prop = nn.Sequential()
#         self.local_prop.add_module(name="local_prop", module=LocalProp(channels=embed_dim, sample_rate=3))

#         # global self attention
#         self.global_rep = nn.Sequential()
#         self.global_rep.add_module(name="attn_gn", module=nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1))
#         self.global_rep.add_module(name="attn", module=SeparableSelfAttention(embed_dim=embed_dim, attn_dropout=attn_dropout))
#         self.global_rep.add_module(name="attn_drop", module=nn.Dropout(dropout))
#         # self.global_rep.add_module(name="local_prop", module=LocalProp(channels=embed_dim, sample_rate=1))

#         # ffn
#         self.ffn = nn.Sequential()
#         self.ffn.add_module(name="ffn_gn", module=nn.GroupNorm(num_channels=2*embed_dim, eps=1e-5, affine=True, num_groups=1))
#         self.ffn.add_module(name="ffn_conv1", module=ConvLayer(in_channels=2*embed_dim, out_channels=ffn_latent_dim, kernel_size=1, stride=1, bias=True, use_norm=False, use_act=True))
#         self.ffn.add_module(name="ffn_drop1", module=nn.Dropout(dropout))
#         self.ffn.add_module(name="ffn_conv2", module=ConvLayer(in_channels=ffn_latent_dim, out_channels=in_channels, kernel_size=1, stride=1, bias=True, use_norm=False, use_act=False))
#         self.ffn.add_module(name="ffn_drop2", module=nn.Dropout(dropout))

#         self.patch_w = patch_w
#         self.patch_h = patch_h

#     def unfolding(self, X: Tensor) -> Tuple[Tensor, Dict]:
#         """
#         将图像变成3*3的一叠
#         """
#         patch_w, patch_h = self.patch_w, self.patch_h
#         patch_area = patch_w * patch_h
#         batch_size, in_channels, orig_h, orig_w = X.shape

#         new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
#         new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

#         interpolate = False
#         if new_w != orig_w or new_h != orig_h:
#             # Note: Padding can be done, but then it needs to be handled in attention function.
#             X = F.interpolate(X, size=(new_h, new_w), mode="bilinear", align_corners=False)
#             interpolate = True

#         # number of patches along width and height
#         num_patch_w = new_w // patch_w  # n_w
#         num_patch_h = new_h // patch_h  # n_h
#         num_patches = num_patch_h * num_patch_w  # N

#         # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
#         X = X.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
#         # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
#         X = X.transpose(1, 2)
#         # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, p_h, p_w] where P = p_h * p_w and N = n_h * n_w
#         X = X.reshape(batch_size, in_channels, num_patches, patch_h, patch_w)
#         # [B, C, N, p_h, p_w] -> [B, N, C, p_h, p_w]
#         X = X.transpose(1, 2)
#         # [B, N, C, p_h, p_w] -> [BN, C, p_h, p_w]
#         X = X.reshape(batch_size * num_patches, in_channels, patch_h, patch_w)
#         X = self.local_agg(X)
#         _,_,agg_patch_h, agg_patch_w = X.shape
#         agg_patch_area = agg_patch_h * agg_patch_w
#         #  -> [B, N, C, P]
#         X = X.reshape(batch_size, num_patches, in_channels, agg_patch_area)
#         #  -> [B, C, N, P]
#         X = X.transpose(1, 2)
#         #  -> [B, C, P, N]
#         X = X.transpose(2, 3)

#         info_dict = {
#             "orig_size": (orig_h, orig_w),
#             "batch_size": batch_size,
#             "interpolate": interpolate,
#             "total_patches": num_patches,
#             "num_patches_w": num_patch_w,
#             "num_patches_h": num_patch_h,
#             "agg_patches_w": agg_patch_w,
#             "agg_patches_h": agg_patch_h
#         }

#         return X, info_dict

#     def folding(self, X: Tensor, info_dict: Dict) -> Tensor:
#         n_dim = X.dim()
#         # [B, C, P, N]
#         assert n_dim == 4, "Tensor should be of shape [B, C, P, N]. Got: {}".format(
#             X.shape
#         )
        
#         num_patch_h = info_dict["num_patches_h"]
#         num_patch_w = info_dict["num_patches_w"]
#         agg_patch_h = info_dict["agg_patches_h"]
#         agg_patch_w = info_dict["agg_patches_w"]
#         batch_size, channels, agg_patch_area, num_patches = X.shape

#         # [B, C, P, N] -> [B, C, N, P]
#         X = X.transpose(2, 3)
#         # [B, C, N, P] -> [B, N, C, P]
#         X = X.transpose(1, 2)
#         # [B, N, C, P] -> [BN, C, p_h, p_w]
#         X = X.reshape(batch_size*num_patches, channels, agg_patch_h, agg_patch_w)
#         # local prop    
#         X = self.local_prop(X)
#         # [BN, C, p_h, p_w] -> [B, N, C, p_h, p_w]
#         X = X.reshape(batch_size, num_patches, channels, self.patch_h, self.patch_w)
#         # [B, N, C, p_h, p_w] -> [B, C, N, p_h, p_w]
#         X = X.transpose(1, 2)
#         # [B, C, N, p_h, p_w] -> [B * C * n_h, n_w, p_h, p_w]
#         X = X.reshape(batch_size*channels*num_patch_h, num_patch_w, self.patch_h, self.patch_w)
#         # [B * C * n_h, n_w, p_h, p_w] -> [B * C * n_h, p_h, n_w, p_w]
#         X = X.transpose(1, 2)
#         # [B * C * n_h, p_h, n_w, p_w] -> [B, C, H, W]
#         X = X.reshape(batch_size, channels, num_patch_h*self.patch_h, num_patch_w*self.patch_w)
#         if info_dict["interpolate"]:
#             X = F.interpolate(
#                 X,
#                 size=info_dict["orig_size"],
#                 mode="bilinear",
#                 align_corners=False,
#             )    
#         return X
    
#     def forward(self, X):
#         # input
#         res = X
#         X = self.input_rep(X)
#         # local 
#         local = self.local_rep(X)
#         # global 
#         # [B, C, H, W] --> [B, C, P, N], 此处要进行分patch
#         globals, info_dict = self.unfolding(X)
#         # global attn
#         globals = self.global_rep(globals)
#         # [B, C, P, N] --> [B, C, H, W], 此处要进行patch还原 
#         globals = self.folding(globals, info_dict)
#         # ffn, 混合全局和局部特征
#         local2global = torch.sigmoid(local)
#         global2local = torch.sigmoid(globals)
#         local = local * global2local
#         globals = globals * local2global
#         res = res + self.ffn(torch.cat((local, globals), dim=1))
#         return res


# 无用的代码
# class ConvAttnFFN(nn.Module):
#     def __init__(self, embed_dim, ffn_latent_dim, dropout=0, attn_dropout=0):
#         super().__init__()
#         self.pre_norm_attn = nn.Sequential(
#             nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
#             LinearSelfAttention(embed_dim, attn_dropout),
#             nn.Dropout(dropout)
#         )
#         self.pre_norm_ffn = nn.Sequential(
#             nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
#             conv_2d(embed_dim, ffn_latent_dim, kernel_size=1, stride=1, bias=True, norm=False, act=True),
#             nn.Dropout(dropout),
#             conv_2d(ffn_latent_dim, embed_dim, kernel_size=1, stride=1, bias=True, norm=False, act=False),
#             nn.Dropout(dropout)
#         )
    
#     def forward(self, x):
#         # self attention
#         x = x + self.pre_norm_attn(x)
#         # Feed Forward network
#         x = x + self.pre_norm_ffn(x)
#         return x


if __name__== "__main__" :
    X = torch.rand(2, 64, 64, 64)   # B. C. H. W
    # model = MyTransformerBlock(16, 64, 8)
    model = MyTransformerEncoder(64, 16)
    print(model(X).shape)