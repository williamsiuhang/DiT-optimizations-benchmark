import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp # type: ignore # type: ignore

class FocusedMQAttention(Attention):
    """
    A modified attention layer that uses a focused attention to decouple Q and K, and a single K-V repeated across heads to reduce complexity and compute.
    References: Ainslie et al: https://arxiv.org/abs/2305.13245; Han et al: https://arxiv.org/abs/2308.00442
    """
    def __init__(self, dim, num_heads: int = 8, p = 3, **kwargs):
        super().__init__(dim, num_heads, **kwargs)    
        self.wq = nn.Linear(dim, dim,  bias=True)
        self.wkv = nn.Linear(dim, self.head_dim * 2,  bias=True)
        self.fused_attn = False
        self.dwc = nn.Conv2d(in_channels=self.head_dim, out_channels=self.head_dim, kernel_size=(3, 3), padding=1, groups=self.head_dim)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(dim, dim, bias=True)
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
        kv = k.transpose(-2, -1) @ v
        x = q @ kv # B, num_heads, N, head_dim

        # Add back DWC on v
        latent_side = int(N ** 0.5)
        sq_v = v.permute(0, 1, 3, 2).reshape(B, self.head_dim, latent_side, latent_side) # B, head_dim, H, W
        v_dwc = self.dwc(sq_v).reshape(B, self.head_dim, N).transpose(-2, -1).unsqueeze(1) # B, 1, N, head_dim

        x = x + v_dwc
        # Project out
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

def main():
    # Example input
    x = torch.randn(2, 16, 384)  # (batch_size, num_patches, embed_dim)
    
    # Create the modified attention layer
    attn_layer = FocusedMQAttention(dim=384, num_heads=6, qkv_bias=True, p=3)
    
    # Forward pass
    output = attn_layer(x)
    
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()