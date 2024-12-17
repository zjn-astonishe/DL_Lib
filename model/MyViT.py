import torch
import math
import torch.nn.functional as F
import torch.nn as nn

from typing import Tuple, Dict
from torch import Tensor
from model.MobileNet import ConvLayer, InvertedResidual
from model.MyTransformer import MyTransformerEncoder
from model.EdgeViT import LocalAgg, LocalProp, LocalAgg1
from model.MyViTConfig import get_config

class MyViTBlock(nn.Module):
    """
    MyViTBlock = Local prep + Attn prep
    """
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        ffn_latent_dim: int,
        dropout: int = 0,
        attn_dropout: int = 0,
        patch_h: int = 3,
        patch_w: int = 3,
        attn_blocks: int = 3,
        conv_ksize: int = 3,
    ) -> None:
        
        super().__init__()

        # input rep
        self.input_rep = nn.Sequential()
        self.input_rep.add_module(name="input_dwconv", module=ConvLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                stride=1,
                kernel_size=conv_ksize,
                groups=in_channels,
                use_norm=True,
                use_act=True,
            )
        )
        self.input_rep.add_module(name="input_pwconv", module=ConvLayer(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                use_norm=False,
                use_act=False
            )
        )

        # local rep
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="local_dwconv", module=ConvLayer(
                in_channels=embed_dim,
                out_channels=embed_dim,
                stride=1,
                kernel_size=conv_ksize,
                groups=embed_dim,
                use_norm=False,
                use_act=True,
            )
        )

        # local agg for attn
        self.local_agg = nn.Sequential()
        self.local_agg.add_module(name="local_agg", module=LocalAgg1(embed_dim))
        
        # local prop for attn
        self.local_prop = nn.Sequential()
        self.local_prop.add_module(name="local_prop", module=LocalProp(channels=embed_dim, sample_rate=3))

        # global self attention
        self.global_rep = nn.Sequential()
        for i in range(attn_blocks):
            self.global_rep.add_module(name=f'MyTransformerEncoder_{i}', module=MyTransformerEncoder(embed_dim=embed_dim, ffn_latent_dim=ffn_latent_dim, dropout=dropout, attn_drop=attn_dropout))
        self.global_rep.add_module(name='MyTransformerEncoderLayerNorm2D', module=nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1))

        # fusion
        self.fusion = nn.Sequential()
        self.fusion.add_module(name="fusion", module=ConvLayer(in_channels=2*embed_dim, out_channels=in_channels, kernel_size=1, stride=1, use_act=False, use_norm=True))
        self.fusion.add_module(name="fusion_drop", module=nn.Dropout(dropout))

        self.patch_w = patch_w
        self.patch_h = patch_h

    def unfolding(self, X: Tensor) -> Tuple[Tensor, Dict]:
        """
        将图像变成3*3的一叠
        """
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = X.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            X = F.interpolate(X, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        X = X.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        X = X.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, p_h, p_w] where P = p_h * p_w and N = n_h * n_w
        X = X.reshape(batch_size, in_channels, num_patches, patch_h, patch_w)
        # [B, C, N, p_h, p_w] -> [B, N, C, p_h, p_w]
        X = X.transpose(1, 2)
        # [B, N, C, p_h, p_w] -> [BN, C, p_h, p_w]
        X = X.reshape(batch_size * num_patches, in_channels, patch_h, patch_w)
        X = self.local_agg(X)
        _,_,agg_patch_h, agg_patch_w = X.shape
        agg_patch_area = agg_patch_h * agg_patch_w
        #  -> [B, N, C, P]
        X = X.reshape(batch_size, num_patches, in_channels, agg_patch_area)
        #  -> [B, C, N, P]
        X = X.transpose(1, 2)
        #  -> [B, C, P, N]
        X = X.transpose(2, 3)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
            "agg_patches_w": agg_patch_w,
            "agg_patches_h": agg_patch_h
        }

        return X, info_dict

    def folding(self, X: Tensor, info_dict: Dict) -> Tensor:
        n_dim = X.dim()
        # [B, C, P, N]
        assert n_dim == 4, "Tensor should be of shape [B, C, P, N]. Got: {}".format(
            X.shape
        )
        
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]
        agg_patch_h = info_dict["agg_patches_h"]
        agg_patch_w = info_dict["agg_patches_w"]
        batch_size, channels, agg_patch_area, num_patches = X.shape

        # [B, C, P, N] -> [B, C, N, P]
        X = X.transpose(2, 3)
        # [B, C, N, P] -> [B, N, C, P]
        X = X.transpose(1, 2)
        # [B, N, C, P] -> [BN, C, p_h, p_w]
        X = X.reshape(batch_size*num_patches, channels, agg_patch_h, agg_patch_w)
        # local prop    
        X = self.local_prop(X)
        # [BN, C, p_h, p_w] -> [B, N, C, p_h, p_w]
        X = X.reshape(batch_size, num_patches, channels, self.patch_h, self.patch_w)
        # [B, N, C, p_h, p_w] -> [B, C, N, p_h, p_w]
        X = X.transpose(1, 2)
        # [B, C, N, p_h, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        X = X.reshape(batch_size*channels*num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B * C * n_h, n_w, p_h, p_w] -> [B * C * n_h, p_h, n_w, p_w]
        X = X.transpose(1, 2)
        # [B * C * n_h, p_h, n_w, p_w] -> [B, C, H, W]
        X = X.reshape(batch_size, channels, num_patch_h*self.patch_h, num_patch_w*self.patch_w)
        if info_dict["interpolate"]:
            X = F.interpolate(
                X,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )    
        return X
    
    def forward(self, X):
        # input
        res = X
        X = self.input_rep(X)
        # local 
        local = self.local_rep(X)
        # global 
        # [B, C, H, W] --> [B, C, P, N], 此处要进行分patch
        globals, info_dict = self.unfolding(X)
        # global encoder
        globals = self.global_rep(globals)
        # [B, C, P, N] --> [B, C, H, W], 此处要进行patch还原 
        globals = self.folding(globals, info_dict)
        # fusion, 混合全局和局部特征
        local2global = torch.sigmoid(local)
        global2local = torch.sigmoid(globals)
        local = local * global2local
        globals = globals * local2global
        res = res + self.fusion(torch.cat((local, globals), dim=1))
        return res

class MyViT(nn.Module):
    """
    组合模型
    """

    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super().__init__()

        image_channels = 3
        out_channels = 16

        # depth wise
        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=image_channels,
            kernel_size=3,
            stride=2,
            groups=image_channels
        )
        # project wise
        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])

        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=num_classes))

        # weight init
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "myvit")
        if block_type.lower() == "myvit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        """
        插入倒残差块
        """
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels
        # 这里的input_channel指的是下一层的
        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        embed_dim = cfg["transformer_channels"]
        ffn_latent_dim = cfg.get("ffn_dim")
        # num_heads = cfg.get("num_heads", 4)
        # head_dim = transformer_dim // num_heads

        # if transformer_dim % head_dim != 0:
        #     raise ValueError("Transformer input dimension should be divisible by head dimension. "
        #                      "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MyViTBlock(
            in_channels=input_channel,
            embed_dim=embed_dim,
            ffn_latent_dim=ffn_latent_dim,
            attn_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 3),
            patch_w=cfg.get("patch_w", 3),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            # head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)

        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x
    
def my_vit_xx_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
    config = get_config("xx_small")
    m = MyViT(config, num_classes=num_classes)
    return m


def mobile_vit_x_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt
    config = get_config("x_small")
    m = MyViT(config, num_classes=num_classes)
    return m


def mobile_vit_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
    config = get_config("small")
    m = MyViT(config, num_classes=num_classes)
    return m


if __name__ == '__main__':
    # model = mobile_vit_xx_small(num_classes=10)
    model = my_vit_xx_small(num_classes=10)
    X = torch.rand(2, 3, 224, 224)   # B. C. H. W
    # model = MyViTBlock(16, 64, 32)
    # print("MyViTBlock")
    print(model(X).shape)
