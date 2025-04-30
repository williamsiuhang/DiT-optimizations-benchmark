import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp # type: ignore # type: ignore

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
        q = q_norm / qp_norm * qp

        k = self.relu(k)
        kp = torch.pow(k, self.p)
        k_norm = torch.linalg.matrix_norm(k).reshape(k.shape[0], k.shape[1], 1, 1)
        kp_norm = torch.linalg.matrix_norm(kp).reshape(kp.shape[0], kp.shape[1], 1, 1)    
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

def main():
    # Example input
    x = torch.randn(2, 16, 384)  # (batch_size, num_patches, embed_dim)
    
    # Create the modified attention layer
    attn_layer = FocusedGQAttention(dim=384, num_heads=6, qkv_bias=True, p=3, grouping=3)
    
    # Forward pass
    output = attn_layer(x)
    
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()