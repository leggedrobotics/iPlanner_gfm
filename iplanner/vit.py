import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ViTFeatureExtractor(nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
        self.vit.head = nn.Identity()  # Remove classification head
        self.patch_size = 16
        if freeze_backbone:
            for param in self.vit.blocks.parameters():
                param.requires_grad = False
    
    def interpolate_pos_embed(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.vit.pos_embed.shape[1] - 1 # Number of original patches
        class_pos_embed =  self.vit.pos_embed[:, 0]
        patch_pos_embed =  self.vit.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def forward(self, x):

        B, _, h, w = x.shape

        # Forward pass
        x = self.vit.patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.interpolate_pos_embed(x, h, w)

        x = self.vit.pos_drop(x)

        # Pass through transformer layers
        for blk in self.vit.blocks:
            x = blk(x)
        
        # Normalize and reshape output
        x = self.vit.norm(x)
        x = x[:, 1:].reshape(B, -1, h // self.patch_size, w // self.patch_size)  # Remove CLS token

        return x

class Dinov2FeatureExtractor(ViTFeatureExtractor):
    
    def __init__(self, freeze_backbone=False, pretrained_ckpt = None):
        super().__init__()
        # self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
        # self.vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
        self.vit = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        
        if pretrained_ckpt is not None:
            ckpt = torch.load(pretrained_ckpt)
            new_ckpt = {k.replace('backbone.', ''): v for k, v in ckpt['teacher'].items()}
            self.vit.load_state_dict(new_ckpt, strict=False)

        self.vit.head = nn.Identity()  # Remove classification head
        self.patch_size = 16
        if freeze_backbone:
            for param in self.vit.blocks.parameters():
                param.requires_grad = False
    