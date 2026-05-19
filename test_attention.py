import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

# --- 1. Mock Globals & Helpers from vit.py ---
NUM_FRAMES = 2
NUM_PATCHES = 4


def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


# --- 2. Original Implementation ---
class OriginalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # Removed .to('cuda') for CPU testing compatibility
        self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES)

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply causal mask
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# --- 3. Refactored Implementation (FlashAttention) ---
class FlashAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES)

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Convert the 1s and 0s mask to a Boolean mask required by SDPA
        attn_mask = self.bias[:, :, :T, :T] == 1
        dropout_p = self.dropout.p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False
        )

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# --- 4. Equivalence Test ---
def test_equivalence():
    # Setup dimensions
    dim = 384
    heads = 6
    dim_head = 64
    seq_len = NUM_FRAMES * NUM_PATCHES
    batch_size = 2

    print("Initializing models...")
    orig_attn = OriginalAttention(dim, heads=heads, dim_head=dim_head)
    flash_attn = FlashAttention(dim, heads=heads, dim_head=dim_head)

    # Share exact weights
    flash_attn.load_state_dict(orig_attn.state_dict())

    # Set to evaluation mode
    orig_attn.eval()
    flash_attn.eval()

    # Create a random dummy input tensor
    x = torch.randn(batch_size, seq_len, dim)

    with torch.no_grad():
        out_orig = orig_attn(x)
        out_flash = flash_attn(x)

    # Compare outputs
    max_diff = torch.abs(out_orig - out_flash).max().item()
    is_close = torch.allclose(out_orig, out_flash, atol=1e-5)

    print(f"Max Absolute Difference: {max_diff:.8f}")
    if is_close:
        print("Result: SUCCESS - The implementations are numerically equivalent.")
    else:
        print("Result: FAILED - Outputs diverge beyond the acceptable tolerance.")


if __name__ == "__main__":
    test_equivalence()