import torch
import torch.nn as nn
import math
from torch import nn, Tensor
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# linear attention
class LinearAttention(nn.Module):
    def __init__(self, dim, *, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
            (q, k, v),
        )

        q = q * self.scale
        q, k = q.softmax(dim=-1), k.softmax(dim=-2)

        if exists(mask):
            k.masked_fill_(mask, 0.0)

        context = einsum("b n d, b n e -> b d e", q, k)
        out = einsum("b d e, b n d -> b n e", context, v)
        out = rearrange(out, " (b h) n d -> b n (h d)", h=h)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_linear_attn: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.use_linear_attn = use_linear_attn

        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)
        self.linear_attn = LinearAttention(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.ffn = FeedForward(dim, dim, ff_mult, *args, **kwargs)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_linear_attn:
            x = self.linear_attn(x)
            x = self.norm(x)
            x = self.ffn(x)
        else:
            x, _, _ = self.attn(x)
            x = self.norm(x)
            x = self.ffn(x)
        return x

#### SOTA CSLR ==========================================

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :].to(x.device)

# Conformer Block
class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # FIX: Move norm out
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 2*d_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (T, B, D)
        residual = x

        # FFN1
        x = x + 0.5 * self.ffn1(x)

        # MHSA
        x = x + self.self_attn(x, x, x, need_weights=False)[0]

        # Conv path
        x_conv = self.conv_norm(x)  # (T, B, D)
        x_conv = x_conv.transpose(0,1).transpose(1,2)  # (B, D, T)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1,2).transpose(0,1)  # (T, B, D)

        x = x + x_conv

        # FFN2
        x = x + 0.5 * self.ffn2(x)

        return self.norm(x)

# Full SOTA CSLR model
class SOTA_CSLR(nn.Module):
    '''
    This is my Proprossed best model for SI task
        - paper model name: `Signer-Invariant Conformer`
    '''
    def __init__(self, vocab_size):
        super().__init__()
        self.input_size = 86 * 2
        self.output_size = vocab_size
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(self.input_size, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),  # stride=2
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 768, kernel_size=3, stride=2, padding=1),  # stride=2
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Conv1d(768, 1024, kernel_size=3, stride=2, padding=1),  # stride=2
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.proj = nn.Linear(1024, 512)
        self.pos_enc = PositionalEncoding(512)

        self.conformers = nn.ModuleList([
            ConformerBlock(512, n_heads=8, ff_mult=4, dropout=0.3)
            for _ in range(6)  # 6 blocks, can tune 6~12
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, self.output_size)
        )

    def forward(self, x):
        # Accept both:
        #   (B, T, 86, 2)  -> flatten -> (B, T, 172)
        #   (B, T, 172)    -> keep
        if x.dim() == 4:
            x = x.view(x.size(0), x.size(1), -1)  # (B, T, 86*2)
        elif x.dim() != 3:
            raise RuntimeError(f"Unexpected input shape: {x.shape}")
    
        # x: (B, T, 86*2)
        x = x.permute(0, 2, 1)         # (B, C, T)
        x = self.temporal_encoder(x)   # (B, C, T')
        x = x.permute(0, 2, 1)         # (B, T', C)
        x = self.proj(x)               # (B, T', 512)
        x = self.pos_enc(x)            # (B, T', 512)
        x = x.transpose(0, 1)          # (T', B, 512)
    
        for block in self.conformers:
            x = block(x)
    
        x = x.transpose(0, 1)          # (B, T', 512)
        x = self.classifier(x)         # (B, T', vocab_size+2)
        return x