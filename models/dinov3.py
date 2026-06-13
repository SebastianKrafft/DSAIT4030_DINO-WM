import torch.nn as nn
import timm


class DinoV3Encoder(nn.Module):
    def __init__(
        self,
        name="vit_small_patch16_dinov3.lvd1689m",
        img_size=224,
    ):
        super().__init__()
        self.name = name
        self.base_model = timm.create_model(name, pretrained=True, img_size=img_size)
        self.emb_dim = self.base_model.num_features
        self.num_patches = self.base_model.patch_embed.num_patches
        self.latent_ndim = 2
        self.patch_size = self.base_model.patch_embed.patch_size[0]

    def forward(self, x):
        tokens = self.base_model.forward_features(x)
        return tokens[:, self.base_model.num_prefix_tokens :]
