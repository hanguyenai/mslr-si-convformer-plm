import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange

# ============================================================
# Positional Encoding
# ============================================================
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


# ============================================================
# Conformer Block 
# ============================================================
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
        x = x + 0.5 * self.ffn1(x)
        x = x + self.self_attn(x, x, x, need_weights=False)[0]

        x_conv = self.conv_norm(x)
        x_conv = x_conv.transpose(0,1).transpose(1,2)  # (B, D, T)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1,2).transpose(0,1)  # (T, B, D)
        x = x + x_conv

        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


# ============================================================
# PLM Decoder - Permutation Language Model Decoder
# ============================================================
class PLMDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder with Permutation Language Modeling (PLM).
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, num_layers=2, 
                 dropout=0.1, max_len=200, num_perms=6):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_perms = num_perms  # K permutations per training example
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=False  # (T, B, D)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)
        
    def generate_permutation_mask(self, seq_len, perm, device):
        order = torch.zeros(seq_len, dtype=torch.long, device=device)
        order[perm] = torch.arange(seq_len, device=device)

        mask = order.unsqueeze(0) >= order.unsqueeze(1)  # (seq_len, seq_len)
        mask = mask.fill_diagonal_(True)
        
        return mask  # True = blocked
    
    def forward(self, encoder_output, targets, target_lengths=None):
        B, L = targets.shape
        device = targets.device
        
        tgt_emb = self.token_embedding(targets)  # (B, L, D)
        tgt_emb = self.pos_encoding(tgt_emb)      # Add positional encoding (canonical order)
        tgt_emb = tgt_emb.transpose(0, 1)          # (L, B, D)
        
        total_loss = 0.0
        
        for k in range(self.num_perms):
            perm = torch.randperm(L, device=device)
            
            attn_mask = self.generate_permutation_mask(L, perm, device)
            
            tgt_key_padding_mask = None
            if target_lengths is not None:
                tgt_key_padding_mask = torch.arange(L, device=device).unsqueeze(0) >= target_lengths.unsqueeze(1)
            
            decoder_output = self.transformer_decoder(
                tgt=tgt_emb,
                memory=encoder_output,
                tgt_mask=attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )  # (L, B, D)
            
            decoder_output = self.norm(decoder_output)
            logits = self.output_proj(decoder_output)  # (L, B, vocab_size)
            logits = logits.permute(1, 0, 2)           # (B, L, vocab_size)
            
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=0,  # ignore padding
                reduction='mean'
            )
            
            total_loss += loss
        
        # Average over K permutations (Equation 7 from paper)
        plm_loss = total_loss / self.num_perms
        
        return plm_loss


# ============================================================
# SOTA_CSLR with PLM Auxiliary Decoder
# ============================================================
class SOTA_CSLR_PLM(nn.Module):
    """
    Signer-Invariant Conformer + PLM Auxiliary Decoder
    
    Architecture:
        Input -> Temporal Encoder -> Conformer Blocks -> 
            ├── CTC Head (main loss)
            └── PLM Decoder (auxiliary loss)
    
    Training: L_total = L_ctc + λ * L_plm
    Inference: Chỉ dùng CTC head (decoder bị bỏ)
    """
    def __init__(self, vocab_size, plm_weight=0.5, num_perms=6):
        super().__init__()
        self.input_size = 86 * 2
        self.output_size = vocab_size
        self.plm_weight = plm_weight
        
        # ===== Encoder =====
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(self.input_size, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 768, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Conv1d(768, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.proj = nn.Linear(1024, 512)
        self.pos_enc = PositionalEncoding(512)

        self.conformers = nn.ModuleList([
            ConformerBlock(512, n_heads=8, ff_mult=4, dropout=0.3)
            for _ in range(6)
        ])
        
        # ===== CTC Head =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, self.output_size)
        )
        
        # ===== PLM Decoder =====
        self.plm_decoder = PLMDecoder(
            vocab_size=vocab_size,
            d_model=512,
            n_heads=8,
            num_layers=2,
            dropout=0.2,
            max_len=200,
            num_perms=num_perms
        )

    def encode(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), x.size(1), -1)
        elif x.dim() != 3:
            raise RuntimeError(f"Unexpected input shape: {x.shape}")
    
        x = x.permute(0, 2, 1)         # (B, C, T)
        x = self.temporal_encoder(x)    # (B, C, T')
        x = x.permute(0, 2, 1)         # (B, T', C)
        x = self.proj(x)               # (B, T', 512)
        x = self.pos_enc(x)            # (B, T', 512)
        x = x.transpose(0, 1)          # (T', B, 512)
    
        for block in self.conformers:
            x = block(x)
        
        return x  # (T', B, 512)
    
    def forward(self, x, targets=None, target_lengths=None):
        # Encoder
        enc_out = self.encode(x)  # (T', B, 512)
        
        # CTC Head
        ctc_logits = enc_out.transpose(0, 1)    # (B, T', 512)
        ctc_logits = self.classifier(ctc_logits)  # (B, T', vocab_size)
        
        # PLM Decoder
        plm_loss = None
        if targets is not None and self.training:
            plm_loss = self.plm_decoder(enc_out, targets, target_lengths)
        
        return ctc_logits, plm_loss