# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp # type: ignore # type: ignore


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                         Modified Attention Layers                             #
#################################################################################

class PlainAttention(Attention):
    """
    Attention layer that does not used scaled-dot-product-attention (for computing GFLOPS)
    """
    def __init__(self, dim, num_heads: int = 8, **kwargs):
        super().__init__(dim, num_heads, **kwargs)    
        self.fused_attn = False

class MediatedAttention(Attention):
    """
    A modified attention layer that uses mediator tokens to reduce the size of attention matrix calculations.
    Does not use fused attention which would typically be used in CUDA -- may not see speed up on GPU.
    Reference: Pu et al. (2024) https://arxiv.org/abs/2408.05710
    """
    def __init__(self, dim, num_heads: int = 8, **kwargs):
        self.mediator_dim = kwargs.pop('mediator_dim', 4)
        super().__init__(dim, num_heads, **kwargs)    
        patch_size = int(self.mediator_dim ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.mediate = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        self.fused_attn = False

    def forward(self, x):
        B, N, C = x.shape
        latent_side = int(N ** 0.5)
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 3, 1)
        q, k, v = qkv.unbind(0)  #B, C, N
        sq_v = v.reshape(B, C, latent_side, latent_side) # B, C, H, W

        # Make mediator tokens t
        sq_q = q.reshape(B, C, latent_side, latent_side) #Square q tokens
        t = self.mediate(sq_q).reshape(B, C, self.mediator_dim).transpose(-2, -1) 
        t = t.reshape(B, self.mediator_dim, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, H, M, C/H

        # Shape into heads
        q = q.reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)  # B, H, N, C/H
        k = k.reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)  # B, H, N, C/H
        v = v.reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)  # B, H, N, C/H
        
        # Compute mediator on k and v
        k = k * self.scale
        tk = t @ k.transpose(-2, -1)
        tk = self.softmax(tk)
        vmed = tk @ v

        # compute q on mediator
        q = q * self.scale
        qt = q @ t.transpose(-2, -1)
        qt = self.softmax(qt)
        qt = self.attn_drop(qt)
        x = qt @ vmed
        x = x.transpose(1, 2).reshape(B, N, C)

        # Add depthwise convolution and rescale
        v_dwc = self.dwc(sq_v).reshape(B, C, N).transpose(-2, -1) # B, C, N
        x = x + v_dwc

        # Project out
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class ShallowAttention(Attention):
    """
    A modified attention layer that reduces the depth of attention matrix calculations (recognizing that depth >> N tokens for small images).
    Instead of assuming each projection matrix W to query, key and value is of size (N, N), we assume it is of size (N, M) where M < N.
    """
    def __init__(self, dim, num_heads, compression, qkv_bias = True,
            proj_bias= True,
            attn_drop= 0,
            proj_drop= 0):
        assert dim % (num_heads * compression) == 0, 'dim should be divisible by num_heads'
        super().__init__(dim, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.head_dim = dim // (num_heads * compression)
        inner_dim = dim // compression
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(inner_dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        
class FocusedMQAttention(nn.Module):
    """
    A modified attention layer that uses a focused attention to decouple Q and K, and a single K-V repeated across heads to reduce complexity and compute.
    References: Ainslie et al: https://arxiv.org/abs/2305.13245; Han et al: https://arxiv.org/abs/2308.00442
    """
    def __init__(self, dim, num_heads: int = 8, p = 3, **kwargs):
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        super().__init__()
        attn_drop = kwargs.pop('attn_drop', 0.0)
        proj_drop = kwargs.pop('proj_drop', 0.0)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.wq = nn.Linear(dim, dim,  bias=True)
        self.wkv = nn.Linear(dim, self.head_dim * 2,  bias=True)
        self.dwc = nn.Conv2d(in_channels=self.head_dim, out_channels=self.head_dim, kernel_size=(3, 3), padding=1, groups=self.head_dim)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.p = 3

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k, v = self.wkv(x).reshape(B, N, 2, 1, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0) # B, 1, N, C/H

        # Focusing function applied to Q and K, p=3 (cubic)
        q = self.relu(q)
        qp = torch.pow(q, self.p)
        q_norm = torch.linalg.matrix_norm(q).reshape(q.shape[0], q.shape[1], 1, 1)
        qp_norm = torch.linalg.matrix_norm(qp).reshape(qp.shape[0], qp.shape[1], 1, 1)
        q = q_norm / qp_norm * qp

        k = self.relu(k)
        kp = torch.pow(k, self.p)
        k_norm = torch.linalg.matrix_norm(k).reshape(k.shape[0], k.shape[1], 1, 1)
        kp_norm = torch.linalg.matrix_norm(kp).reshape(kp.shape[0], kp.shape[1], 1, 1)    
        k = k_norm / kp_norm * kp
        
        # Compute focused attention starting with k@v
        kv = k.transpose(-2, -1).contiguous() @ v
        x = q.contiguous() @ kv # B, num_heads, N, head_dim

        # Add back DWC on v
        latent_side = int(N ** 0.5)
        sq_v = v.permute(0, 1, 3, 2).reshape(B, self.head_dim, latent_side, latent_side).contiguous() # B, head_dim, H, W
        v_dwc = self.dwc(sq_v).reshape(B, self.head_dim, N).transpose(-2, -1).unsqueeze(1).contiguous() # B, 1, N, head_dim

        x = x + v_dwc
        # Project out
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class FocusedGQAttention(nn.Module):
    """
    A modified attention layer that uses a focused attention to decouple Q and K, and 2 K-V pairs repeated across heads to reduce complexity and compute.
    References: Ainslie et al: https://arxiv.org/abs/2305.13245; Han et al: https://arxiv.org/abs/2308.00442
    """
    def __init__(self, dim, num_heads: int = 8, p = 3, grouping = 3, **kwargs):
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        super().__init__()
        attn_drop = kwargs.pop('attn_drop', 0.0)
        proj_drop = kwargs.pop('proj_drop', 0.0)
        self.num_heads = num_heads
        self.grouping = grouping
        self.num_kv_heads = self.num_heads // self.grouping
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False
        self.p = p
        
        
        self.wq = nn.Linear(dim, dim,  bias=True)
        self.wkv = nn.Linear(dim, dim // self.grouping * 2,  bias=True)
        self.dwc = nn.Conv2d(in_channels=dim // self.grouping, out_channels=dim // self.grouping, kernel_size=(3, 3), padding=1, groups=dim // self.grouping)
        self.relu = nn.ReLU()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, H, N, C/H
        k, v = self.wkv(x).reshape(B, N, 2, self.num_kv_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0) # B, H_kv, N, C/H

        # Focusing function applied to Q and K, p=3 (cubic)
        q = self.relu(q)
        qp = torch.pow(q, self.p)
        q_norm = torch.linalg.matrix_norm(q).reshape(q.shape[0], q.shape[1], 1, 1)
        qp_norm = torch.linalg.matrix_norm(qp).reshape(qp.shape[0], qp.shape[1], 1, 1)
        if torch.any(qp_norm == 0): 
            qp_norm[qp_norm == 0] = 1e-12
        q = q_norm / qp_norm * qp

        k = self.relu(k)
        kp = torch.pow(k, self.p)
        k_norm = torch.linalg.matrix_norm(k).reshape(k.shape[0], k.shape[1], 1, 1)
        kp_norm = torch.linalg.matrix_norm(kp).reshape(kp.shape[0], kp.shape[1], 1, 1)  
        if torch.any(kp_norm == 0): 
            kp_norm[kp_norm == 0] = 1e-12  
        k = k_norm / kp_norm * kp
        
        # Compute focused attention starting with k@v
        kv = k.transpose(-2, -1) @ v
        x = q @ kv.repeat(1, self.grouping, 1, 1) # B, num_heads, N, head_dim

        # Add back DWC on v
        latent_side = int(N ** 0.5)
        sq_v = v.permute(0, 1, 3, 2).reshape(B, self.num_kv_heads * self.head_dim, latent_side, latent_side) # B, V-dim, H, W
        v_dwc = self.dwc(sq_v).reshape(B, self.num_kv_heads, self.head_dim, N).transpose(-2, -1) # B, 1, N, head_dim

        x = x + v_dwc.repeat(1, self.grouping, 1, 1) # B, num_heads, N, head_dim
        # Project out
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        att = block_kwargs.pop('att', None)
        mediator_dim = block_kwargs.pop('mediator_dim', 4)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if att == "plain":
            self.attn = PlainAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        elif att == "med":
            self.attn = MediatedAttention(hidden_size, num_heads=num_heads, qkv_bias=True, mediator_dim=mediator_dim, **block_kwargs)
        elif att == "shallow":
            self.attn = ShallowAttention(hidden_size, num_heads=num_heads, qkv_bias=True, compression=2, **block_kwargs)
        elif att == "fmq":
            self.attn = FocusedMQAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        elif att == "fgq":
            self.attn = FocusedGQAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        else:
            #print(f"Supplied att is not a valid attention type. Using default attention.")
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        att = None,
        mediator_dim=4,
        profile=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.profile = profile

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mediator_dim=mediator_dim, mlp_ratio=mlp_ratio, att=att) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            if self.profile:
                x = block(x, c)
            else:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)       # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_XS_2(**kwargs):
    return DiT(depth=6, hidden_size=256, patch_size=2, num_heads=4, **kwargs)

def DiT_XS_4(**kwargs):
    return DiT(depth=6, hidden_size=256, patch_size=4, num_heads=4, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiT-XS/2': DiT_XS_2, 'DiT-XS/4': DiT_XS_4
}
