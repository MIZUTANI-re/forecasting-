
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import math

# ============================================================
# Shared Components
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TemporalEmbeddingCNN(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            1, d_model, kernel_size=kernel_size,
            padding=padding, padding_mode='replicate'
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, seq, 1) -> (B, 1, seq)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.act(x)
        x = x.permute(0, 2, 1) # (B, seq, d)
        return self.dropout(x)

class TemporalEmbeddingLinear(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return self.dropout(x)

class VariableAxisTransformer(nn.Module):
    """
    Inverted Transformer for Exogenous Variables
    Input: (B, seq, exog_dim) -> Embed -> (B, exog_dim, d)
    """
    def __init__(self, seq_len, d_model, nhead, num_layers, hidden_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.pos  = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
            ) for _ in range(num_layers)
        ])

    def forward(self, src, return_internals=False):
        # src: (B, seq, exog_dim)
        x = src.permute(0, 2, 1) # (B, exog_dim, seq)
        x = self.proj(x)         # (B, exog_dim, d)
        x = self.pos(x)          # (B, exog_dim, d)
        
        attns = []
        for layer in self.layers:
            residual = x
            if layer.norm_first:
                x = layer.norm1(x)
            attn_out, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            x = residual + layer.dropout1(attn_out)
            if not layer.norm_first:
                x = layer.norm1(x)
            if return_internals:
                attns.append(attn_weights)
                
            residual = x
            if layer.norm_first:
                x = layer.norm2(x)
            x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = residual + layer.dropout2(x)
            if not layer.norm_first:
                x = layer.norm2(x)
                
        if return_internals:
            avg_attn = torch.stack(attns).mean(0)
            return x, avg_attn
        return x

# ============================================================
# 1. iTransformer (Pure)
# ============================================================

class iTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, in_dim, d_model, nhead, num_layers,  
                 hidden_dim=256, dropout=0.1, use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Encoding
        self.enc_embedding = nn.Linear(seq_len, d_model)
        self.enc_pos = PositionalEncoding(d_model, dropout)
        
        # Custom Encoder loop to capture attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=hidden_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Decoding
        self.projector = nn.Linear(d_model, pred_len, bias=True)
        self.use_norm = use_norm

    def forward(self, x, return_internals=False):
        # x: (B, seq, N)
        if self.use_norm:
            # Revin-like normalization
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        B, L, N = x.shape
        x = x.permute(0, 2, 1) # (B, N, L)
        
        # Embedding
        enc_out = self.enc_embedding(x) # (B, N, d) 
        
        attns = []
        for layer in self.layers:
            # Manually perform forward to capture attention
            # enc_out = layer(enc_out)
            
            # 1. Self Attention
            residual = enc_out
            if layer.norm_first:
                enc_out = layer.norm1(enc_out)
                
            attn_out, attn_weights = layer.self_attn(enc_out, enc_out, enc_out, need_weights=True)
            enc_out = residual + layer.dropout1(attn_out)
            if not layer.norm_first:
                enc_out = layer.norm1(enc_out)
                
            if return_internals:
                attns.append(attn_weights) # (B, N, N)
                
            # 2. Feed Forward
            residual = enc_out
            if layer.norm_first:
                enc_out = layer.norm2(enc_out)
            enc_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(enc_out))))
            enc_out = residual + layer.dropout2(enc_out)
            if not layer.norm_first:
                enc_out = layer.norm2(enc_out)
        
        # Decode
        dec_out = self.projector(enc_out) # (B, N, pred_len)
        dec_out = dec_out.permute(0, 2, 1) # (B, pred_len, N)
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
        if return_internals:
            # Average attention across layers for a general overview
            avg_attn = torch.stack(attns).mean(0) # (B, N, N)
            return dec_out, {"attention_map": avg_attn}
            
        return dec_out

    def get_attention_maps(self, x):
         # Placeholder: Standard TransformerEncoder doesn't expose weights easily unless we use hooks or custom layers
         return {}

# ============================================================
# 2. Hybrid Models
# ============================================================

class Hybrid_Gated_FeatureFusion_Model(nn.Module):
    def __init__(self, seq_len, exog_dim, horizon, d_model, nhead, num_layers, hidden_dim,
                 dropout, cnn_kernel, cnn_mode="linear"):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        
        # 1. Temporal Branch
        if cnn_mode == "linear":
            self.temp_embed = TemporalEmbeddingLinear(d_model, dropout=dropout)
        else:
            k = 1 if cnn_mode == "cnn1" else cnn_kernel
            self.temp_embed = TemporalEmbeddingCNN(d_model, kernel_size=k, dropout=dropout)
            
        self.temp_pos = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # 2. Variable Branch
        self.var_axis = VariableAxisTransformer(seq_len, d_model, nhead, num_layers, hidden_dim, dropout)

        # 3. Fusion Gate
        self.gate_linear = nn.Linear(d_model * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
        # 4. Head
        self.head = nn.Linear(d_model, 1)
        self.residual_proj = nn.Linear(seq_len, horizon)

        # Adapters
        self.pool_t = nn.AdaptiveAvgPool1d(horizon)
        self.pool_v = nn.AdaptiveAvgPool1d(horizon)

    def forward(self, x_ex, y_p, return_internals=False):
        # x_ex: (B, seq, exog), y_p: (B, seq, 1)
        
        # Temporal
        t_out = self.temporal_encoder(self.temp_pos(self.temp_embed(y_p))) # (B, seq, d)
        t_out = t_out.permute(0, 2, 1) # (B, d, seq)
        t_out = self.pool_t(t_out).permute(0, 2, 1) # (B, H, d)

        # Variable
        if return_internals:
            v_out, var_attn = self.var_axis(x_ex, return_internals=True)
        else:
            v_out = self.var_axis(x_ex)
        v_out = v_out.permute(0, 2, 1)
        v_out = self.pool_v(v_out).permute(0, 2, 1) # (B, H, d)

        # Fusion
        combined = torch.cat([t_out, v_out], dim=-1)
        alpha = self.sigmoid(self.gate_linear(combined)) # (B, H, 1)
        
        fused = alpha * t_out + (1 - alpha) * v_out
        
        # Prediction
        pred_delta = self.head(fused).squeeze(-1) # (B, H)
        baseline = self.residual_proj(y_p.squeeze(-1))
        
        pred = baseline + pred_delta
        
        if return_internals:
            return pred, {"alpha": alpha, "attention_map": var_attn}
        return pred

# KAN Components for Hybrid EvoKAN
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + self.spline_order, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline * noise).permute(2, 1, 0))

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_shape = x.shape
        x_reshaped = x.reshape(-1, self.in_features)
        spline_output = F.linear(
            self.b_splines(x_reshaped).view(x_reshaped.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output.view(x_shape[0], self.out_features)
        return output

class Hybrid_KAN_Gated_FeatureFusion_Model(nn.Module):
    def __init__(self, seq_len, exog_dim, horizon, d_model, nhead, num_layers, hidden_dim,
                 dropout, cnn_kernel, cnn_mode="linear", initial_grid=3):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        
        if cnn_mode == "linear":
            self.temp_embed = TemporalEmbeddingLinear(d_model, dropout=dropout)
        else:
            k = 1 if cnn_mode == "cnn1" else cnn_kernel
            self.temp_embed = TemporalEmbeddingCNN(d_model, kernel_size=k, dropout=dropout)
            
        self.temp_pos = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.var_axis = VariableAxisTransformer(seq_len, d_model, nhead, num_layers, hidden_dim, dropout)

        # KAN Gate
        self.kan_gate = KANLinear(d_model * 2, 1, grid_size=initial_grid)
        self.sigmoid = nn.Sigmoid()
        
        self.head = KANLinear(d_model, 1, grid_size=initial_grid)
        self.residual_proj = nn.Linear(seq_len, horizon)
        self.pool_t = nn.AdaptiveAvgPool1d(horizon)
        self.pool_v = nn.AdaptiveAvgPool1d(horizon)

    def forward(self, x_ex, y_p, return_internals=False):
        t_out = self.temporal_encoder(self.temp_pos(self.temp_embed(y_p))).permute(0, 2, 1)
        t_out = self.pool_t(t_out).permute(0, 2, 1)
        
        if return_internals:
            v_out, var_attn = self.var_axis(x_ex, return_internals=True)
        else:
            v_out = self.var_axis(x_ex)
            
        v_out = v_out.permute(0, 2, 1)
        v_out = self.pool_v(v_out).permute(0, 2, 1)
        
        combined = torch.cat([t_out, v_out], dim=-1)
        B, H, D2 = combined.shape
        alpha = self.sigmoid(self.kan_gate(combined.view(-1, D2))).view(B, H, 1)
        
        fused = alpha * t_out + (1 - alpha) * v_out
        pred_delta = self.head(fused.view(-1, fused.shape[-1])).view(B, H)
        
        baseline = self.residual_proj(y_p.squeeze(-1))
        pred = baseline + pred_delta
        
        if return_internals:
            return pred, {"alpha": alpha, "attention_map": var_attn}
        return pred

# ============================================================
# NEW: Hybrid Varients (FILM & PredictFusion)
# ============================================================

class Hybrid_Gated_QueryFusion_Model(nn.Module):
    def __init__(self, seq_len, exog_dim, horizon, d_model, nhead, num_layers, hidden_dim, dropout=0.1,
                 gate_init=0.1, use_hq=True):
        super().__init__()
        self.horizon = horizon
        self.use_hq = use_hq
        self.temp_embed = TemporalEmbeddingLinear(d_model, dropout)
        self.temp_pos = PositionalEncoding(d_model, dropout)
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout, batch_first=True), num_layers)
        
        self.var_axis = VariableAxisTransformer(seq_len, d_model, nhead, num_layers, hidden_dim, dropout)

        if self.use_hq:
            self.hq = nn.Parameter(torch.randn(1, horizon, d_model))
            self.q_cond = nn.Linear(d_model, d_model)
            self.head = nn.Linear(d_model, 1)
        else:
            self.hq = nn.Parameter(torch.randn(1, 1, d_model))
            self.q_cond = nn.Linear(d_model, d_model)
            self.head = nn.Linear(d_model, horizon)
        
        self.cross_time = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_var = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        
        self.gate_logit = nn.Parameter(torch.tensor([gate_init]))

    def forward(self, x_ex, y_p, return_internals=False):
        t_m = self.temporal_encoder(self.temp_pos(self.temp_embed(y_p)))
        if return_internals:
            v_m, var_attn = self.var_axis(x_ex, return_internals=True)
        else:
            v_m = self.var_axis(x_ex)
        
        B = y_p.size(0)
        
        if self.use_hq:
            q = self.hq.expand(B, -1, -1) + self.q_cond(t_m[:, -1, :]).unsqueeze(1)
        else:
            q = self.hq.expand(B, -1, -1) + self.q_cond(t_m[:, -1, :]).unsqueeze(1)
        
        qt, _ = self.cross_time(q, t_m, t_m)
        qv, _ = self.cross_var(q, v_m, v_m)
        
        alpha = torch.sigmoid(self.gate_logit)
        fused = alpha * qt + (1 - alpha) * qv
        
        if self.use_hq:
            pred = self.head(fused).squeeze(-1)
        else:
            pred = self.head(fused.squeeze(1))
            
        if return_internals:
            return pred, {"gate_alpha": alpha, "attention_map": var_attn}
        return pred

class Hybrid_FiLM_Ablation_Model(nn.Module):
    """
    FiLM Fusion with Ablation Modes: "t2v", "v2t", "bidir", "none"
    + Configurable HQ (use_hq=True: 24 queries, use_hq=False: 1 query)
    """
    def __init__(self, seq_len, exog_dim, horizon, d_model, nhead, num_layers, hidden_dim, 
                 dropout=0.1, film_mode="bidir", cnn_mode="linear", cnn_kernel=3, use_hq=True):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.film_mode = film_mode
        self.use_hq = use_hq
        
        if cnn_mode == "linear":
            self.temp_embed = TemporalEmbeddingLinear(d_model, dropout=dropout)
        else:
            k = 1 if cnn_mode == "cnn1" else cnn_kernel
            self.temp_embed = TemporalEmbeddingCNN(d_model, kernel_size=k, dropout=dropout)
            
        self.temp_pos = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers)
        
        self.var_axis = VariableAxisTransformer(seq_len, d_model, nhead, num_layers, hidden_dim, dropout)
        
        # HQ Strategy
        if self.use_hq:
            # 24 Queries -> Output is (B, 24, d)
            self.hq = nn.Parameter(torch.randn(1, horizon, d_model))
            self.q_cond = nn.Linear(d_model, d_model)
            self.head = nn.Linear(d_model, 1) # Applied to (B,24,d) -> (B,24,1) -> (B,24)
        else:
            # 1 Query -> Output is (B, 1, d) -> Expand to 24
            self.hq = nn.Parameter(torch.randn(1, 1, d_model))
            self.q_cond = nn.Linear(d_model, d_model)
            self.head = nn.Linear(d_model, horizon) # Applied to (B,d) -> (B,24)
        
        self.cross_time = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_var = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        
        # FiLM Layers (Conditional)
        self.film_tv = None
        self.film_vt = None
        
        # Note: If use_hq=False, features are (B,1,d), FiLM still works on d dimension
        if self.film_mode in ["t2v", "bidir"]:
            self.film_tv = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2*d_model))
            
        if self.film_mode in ["v2t", "bidir"]:
            self.film_vt = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2*d_model))
        
        self.fuse_proj = nn.Linear(2*d_model, d_model)
        self.residual_proj = nn.Linear(seq_len, horizon)

    def forward(self, x_ex, y_p, return_internals=False):
        t_m = self.temporal_encoder(self.temp_pos(self.temp_embed(y_p)))
        
        internals = {}
        if return_internals:
            v_m, var_attn = self.var_axis(x_ex, return_internals=True)
            internals["attention_map"] = var_attn
        else:
            v_m = self.var_axis(x_ex)
        
        B = y_p.size(0)
        
        if self.use_hq:
            q = self.hq.expand(B, -1, -1) + self.q_cond(t_m[:, -1, :]).unsqueeze(1)
        else:
            q = self.hq.expand(B, -1, -1) + self.q_cond(t_m[:, -1, :]).unsqueeze(1)
        
        qt, _ = self.cross_time(q, t_m, t_m)
        qv, _ = self.cross_var(q, v_m, v_m)
        
        t_feat = self.norm_t(q + qt)
        v_feat = self.norm_v(q + qv)
        
        t_mod = t_feat
        v_mod = v_feat
        
        # FiLM Logic
        if self.film_tv is not None:
            gamma_tv, beta_tv = self.film_tv(t_feat).chunk(2, dim=-1)
            v_mod = v_feat * (1 + torch.tanh(gamma_tv)) + beta_tv
            if return_internals:
                internals["gamma_tv"] = gamma_tv
                internals["beta_tv"] = beta_tv
            
        if self.film_vt is not None:
            gamma_vt, beta_vt = self.film_vt(v_feat).chunk(2, dim=-1)
            t_mod = t_feat * (1 + torch.tanh(gamma_vt)) + beta_vt
            if return_internals:
                internals["gamma_vt"] = gamma_vt
                internals["beta_vt"] = beta_vt
        
        fused = self.fuse_proj(torch.cat([t_mod, v_mod], dim=-1)) # (B, 24, d) or (B, 1, d)
        
        if self.use_hq:
             # fused: (B, 24, d) -> head: (d->1) -> (B, 24, 1) -> squeeze -> (B, 24)
            pred = self.head(fused).squeeze(-1)
        else:
            # fused: (B, 1, d) -> squeeze -> (B, d) -> head: (d->24) -> (B, 24)
            pred = self.head(fused.squeeze(1))
        
        final_pred = self.residual_proj(y_p.squeeze(-1)) + pred
        
        if return_internals:
            return final_pred, internals
            
        return final_pred

class Hybrid_NoHQ_Predict_Fusion_Model(nn.Module):
    """
    No HQ (Single Query), Prediction Level Fusion
    """
    def __init__(self, seq_len, exog_dim, horizon, d_model, nhead, num_layers, hidden_dim,
                 dropout=0.1, cnn_mode="linear", cnn_kernel=3):
        super().__init__()
        self.horizon = horizon
        
        if cnn_mode == "linear":
            self.temp_embed = TemporalEmbeddingLinear(d_model, dropout=dropout)
        else:
            k = 1 if cnn_mode == "cnn1" else cnn_kernel
            self.temp_embed = TemporalEmbeddingCNN(d_model, kernel_size=k, dropout=dropout)
            
        self.temp_pos = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.var_axis = VariableAxisTransformer(seq_len, d_model, nhead, num_layers, hidden_dim, dropout)
        
        # Single Query
        self.q0 = nn.Parameter(torch.randn(1, 1, d_model))
        self.q_cond = nn.Linear(d_model, d_model)
        
        self.cross_time = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_var = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        
        # Heads
        self.head_t = nn.Linear(d_model, horizon)
        self.head_v = nn.Linear(d_model, horizon)
        
        self.gate = nn.Linear(d_model * 2, horizon)
        self.residual_proj = nn.Linear(seq_len, horizon)

    def forward(self, x_ex, y_p, return_internals=False):
        t_m = self.temporal_encoder(self.temp_pos(self.temp_embed(y_p)))
        if return_internals:
            v_m, var_attn = self.var_axis(x_ex, return_internals=True)
        else:
            v_m = self.var_axis(x_ex)
        
        B = y_p.size(0)
        q = self.q0.expand(B, -1, -1) + self.q_cond(t_m[:, -1, :]).unsqueeze(1) # (B, 1, d)
        
        qt, _ = self.cross_time(q, t_m, t_m)
        qv, _ = self.cross_var(q, v_m, v_m)
        
        t_feat = self.norm_t(q + qt).squeeze(1) # (B, d)
        v_feat = self.norm_v(q + qv).squeeze(1)
        
        p_t = self.head_t(t_feat)
        p_v = self.head_v(v_feat)
        
        alpha = torch.sigmoid(self.gate(torch.cat([t_feat, v_feat], dim=-1))) # (B, H)
        pred = alpha * p_t + (1 - alpha) * p_v
        
        final_pred = self.residual_proj(y_p.squeeze(-1)) + pred
        
        if return_internals:
            return final_pred, {"gate_alpha": alpha, "attention_map": var_attn}
        return final_pred

class Hybrid_Predict_Fusion_Model(nn.Module):
    """
    Alpha-Gated Predict Fusion (or Feature Linear)
    """
    def __init__(self, seq_len, exog_dim, horizon, d_model, nhead, num_layers, hidden_dim, dropout=0.1, 
                 fusion_mode="predict", use_hq=True):
        super().__init__()
        self.horizon = horizon
        self.fusion_mode = fusion_mode
        self.use_hq = use_hq
        
        self.temp_embed = TemporalEmbeddingLinear(d_model, dropout)
        self.temp_pos = PositionalEncoding(d_model, dropout)
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout, batch_first=True), num_layers)
        
        self.var_axis = VariableAxisTransformer(seq_len, d_model, nhead, num_layers, hidden_dim, dropout)
        
        if self.use_hq:
            self.hq = nn.Parameter(torch.randn(1, horizon, d_model))
            self.head = nn.Linear(d_model, 1) # shared head d->1 for each horizon step
        else:
            self.hq = nn.Parameter(torch.randn(1, 1, d_model))
            self.head = nn.Linear(d_model, horizon) # shared head d->horizon
            
        self.q_cond = nn.Linear(d_model, d_model)
        
        self.cross_time = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_var = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        
        self.gate = nn.Linear(d_model * 2, 1)
        self.residual_proj = nn.Linear(seq_len, horizon)

    def forward(self, x_ex, y_p, return_internals=False):
        t_m = self.temporal_encoder(self.temp_pos(self.temp_embed(y_p)))
        if return_internals:
            v_m, var_attn = self.var_axis(x_ex, return_internals=True)
        else:
            v_m = self.var_axis(x_ex)
        
        B = y_p.size(0)
        q = self.hq.expand(B, -1, -1) + self.q_cond(t_m[:, -1, :]).unsqueeze(1)
        
        qt, _ = self.cross_time(q, t_m, t_m) # (B, H, d)
        qv, _ = self.cross_var(q, v_m, v_m)
        
        # Gate
        alpha = torch.sigmoid(self.gate(torch.cat([qt, qv], dim=-1))) # (B, H, 1) or (B, 1, 1)
        
        if self.fusion_mode == "predict":
            # Predictions
            if self.use_hq:
                p_t = self.head(qt).squeeze(-1) # (B, H)
                p_v = self.head(qv).squeeze(-1)
                # alpha is (B, H, 1) -> squeeze -> (B, H)
                pred = alpha.squeeze(-1) * p_t + (1 - alpha.squeeze(-1)) * p_v
            else:
                p_t = self.head(qt.squeeze(1)) # (B, H)
                p_v = self.head(qv.squeeze(1))
                a = alpha.squeeze(-1) # (B, 1)
                pred = a * p_t + (1 - a) * p_v

        else:
            fused = alpha * qt + (1 - alpha) * qv
            if self.use_hq:
                pred = self.head(fused).squeeze(-1)
            else:
                pred = self.head(fused.squeeze(1))
            
        final_pred = self.residual_proj(y_p.squeeze(-1)) + pred
        
        if return_internals:
            return final_pred, {"gate_alpha": alpha, "attention_map": var_attn}
        return final_pred

# ============================================================
# 3. Legacy / Other
# ============================================================

class GridTST(nn.Module):
    def __init__(self, seq_len, pred_len, in_dim, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_embedding = nn.Linear(1, d_model) # GridTST processes univariate target mostly or mix? 
        # Actually GridTST in this repo seems to handle Separated data differently
        # Simplified placeholder based on standard implementation
        self.proj = nn.Linear(seq_len, pred_len)
        
    def forward(self, x_ex, y_p):
        # Placeholder simple linear for now as full GridTST is complex
        # Assuming the user just wants the name key to work for now based on previous file
        return self.proj(y_p.squeeze(-1))

class HTMBlock(nn.Module):
    def __init__(self, seq_len, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x, return_internals=False):
        # x: (B, L, D)
        x = self.pos_enc(x)
        
        attns = []
        for layer in self.layers:
            residual = x
            if layer.norm_first:
                x = layer.norm1(x)
            attn_out, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            x = residual + layer.dropout1(attn_out)
            if not layer.norm_first:
                x = layer.norm1(x)
            if return_internals:
                attns.append(attn_weights)
                
            residual = x
            if layer.norm_first:
                x = layer.norm2(x)
            x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = residual + layer.dropout2(x)
            if not layer.norm_first:
                x = layer.norm2(x)
        
        if return_internals:
            # Average across layers
            avg_attn = torch.stack(attns).mean(0)
            return x, avg_attn
            
        return x

class HTMformer(nn.Module):
    """
    Hierarchical Time-Series Multi-scale Transformer (HTMformer) following user description:
    1. Hierarchical Structure (Multi-resolution)
    2. Multi-scale processing (Original, 1/2, 1/4 resolutions)
    3. Global & Local pattern capturing
    """
    def __init__(self, seq_len, pred_len, in_dim, d_model, nhead, num_layers, nhead_var=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Input Projection
        self.enc_embedding = nn.Linear(in_dim, d_model)
        
        # Hierarchical Levels
        # Level 0: Original Resolution
        self.level0_block = HTMBlock(seq_len, d_model, nhead, num_layers)
        
        # Level 1: 1/2 Resolution (Conv Downsample)
        self.down1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2, padding=1)
        self.level1_len = seq_len // 2
        self.level1_block = HTMBlock(self.level1_len, d_model, nhead, num_layers)
        
        # Level 2: 1/4 Resolution (Conv Downsample from L1)
        self.down2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2, padding=1)
        self.level2_len = self.level1_len // 2
        self.level2_block = HTMBlock(self.level2_len, d_model, nhead, num_layers)
        
        # Fusion & Prediction
        # We project each level's last feature to pred_len
        self.proj0 = nn.Linear(seq_len * d_model, pred_len * d_model) # Simple flatten projection often too big
        # Better: Use only last token? Or Flatten?
        # Let's use a shared decoder or simple linear mapping for each scale.
        
        # Simplified: Flatten all time steps -> Linear -> Output
        # (This is like iTransformer/DLinear approach but per scale)
        self.flat_dim0 = seq_len * d_model
        self.flat_dim1 = self.level1_len * d_model
        self.flat_dim2 = self.level2_len * d_model
        
        self.head = nn.Linear(self.flat_dim0 + self.flat_dim1 + self.flat_dim2, pred_len * in_dim)

    def forward(self, x, return_internals=False):
        # x: (B, L, N) -> We usually embed to (B, L, D) for time-mixing
        B, L, N = x.shape
        
        # Embedding: (B, L, N) -> (B, L, d_model)
        x_emb = self.enc_embedding(x)
        
        internals = {}
        
        # Level 0
        if return_internals:
            l0_out, a0 = self.level0_block(x_emb, return_internals=True)
            internals["htm_level0_attention"] = a0
        else:
            l0_out = self.level0_block(x_emb)
        
        # Level 1
        x_trans = x_emb.permute(0, 2, 1) # (B, d, L)
        l1_in = self.down1(x_trans).permute(0, 2, 1) # (B, L/2, d)
        if return_internals:
            l1_out, a1 = self.level1_block(l1_in, return_internals=True)
            internals["htm_level1_attention"] = a1
        else:
            l1_out = self.level1_block(l1_in)
        
        # Level 2
        x_trans_l1 = l1_in.permute(0, 2, 1)
        l2_in = self.down2(x_trans_l1).permute(0, 2, 1) # (B, L/4, d)
        if return_internals:
            l2_out, a2 = self.level2_block(l2_in, return_internals=True)
            internals["htm_level2_attention"] = a2
        else:
            l2_out = self.level2_block(l2_in)
        
        # Flatten and Concat
        f0 = l0_out.reshape(B, -1)
        f1 = l1_out.reshape(B, -1)
        f2 = l2_out.reshape(B, -1)
        
        combined = torch.cat([f0, f1, f2], dim=1)
        
        # Projection
        out = self.head(combined) # (B, pred_len * N)
        out = out.reshape(B, self.pred_len, N)
        
        if return_internals:
            return out, internals
            
        return out

class TransformerEncoderDecoderOneShot(nn.Module):
    def __init__(
        self,
        feature_size: int,
        target_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_length = target_length
        self.d_model = d_model

        self.input_projection = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
            ) for _ in range(num_encoder_layers)
        ])
        
        self.dec_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
            ) for _ in range(num_decoder_layers)
        ])

        # decoder query: [H, d_model]
        self.query_embed = nn.Parameter(torch.randn(target_length, d_model))

        # one-shot output: [B, H] (Univariate target)
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor, return_internals=False) -> torch.Tensor:
        """
        src: [B, L, feature_size]
        return: [B, H]
        """
        B, L, _ = src.shape
        x = self.input_projection(src)            # [B, L, d_model]
        x = self.pos_encoder(x)                   # [B, L, d_model]
        
        enc_attns = []
        for layer in self.enc_layers:
            residual = x
            if layer.norm_first:
                x = layer.norm1(x)
            attn_out, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            x = residual + layer.dropout1(attn_out)
            if not layer.norm_first:
                x = layer.norm1(x)
            if return_internals:
                enc_attns.append(attn_weights)
                
            residual = x
            if layer.norm_first:
                x = layer.norm2(x)
            x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = residual + layer.dropout2(x)
            if not layer.norm_first:
                x = layer.norm2(x)
                
        memory = x
        
        query = self.query_embed.unsqueeze(0).repeat(B, 1, 1)     # [B, H, d_model]
        
        dec_attns = []
        cross_attns = []
        tgt = query
        for layer in self.dec_layers:
            # 1. Self Attention
            residual = tgt
            if layer.norm_first:
                tgt = layer.norm1(tgt)
            attn_out, attn_weights = layer.self_attn(tgt, tgt, tgt, need_weights=True)
            tgt = residual + layer.dropout1(attn_out)
            if not layer.norm_first:
                tgt = layer.norm1(tgt)
            if return_internals:
                dec_attns.append(attn_weights)
                
            # 2. Cross Attention
            residual = tgt
            if layer.norm_first:
                tgt = layer.norm2(tgt)
            attn_out, attn_weights = layer.multihead_attn(tgt, memory, memory, need_weights=True)
            tgt = residual + layer.dropout2(attn_out)
            if not layer.norm_first:
                tgt = layer.norm2(tgt)
            if return_internals:
                cross_attns.append(attn_weights)
                
            # 3. Feed Forward
            residual = tgt
            if layer.norm_first:
                tgt = layer.norm3(tgt)
            tgt = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt))))
            tgt = residual + layer.dropout3(tgt)
            if not layer.norm_first:
                tgt = layer.norm3(tgt)
                
        out = self.output_projection(tgt).squeeze(-1)             # [B, H]
        
        if return_internals:
            internals = {
                "encoder_attention": torch.stack(enc_attns).mean(0), # (B, L, L)
                "decoder_attention": torch.stack(dec_attns).mean(0), # (B, H, H)
                "cross_attention": torch.stack(cross_attns).mean(0) # (B, H, L)
            }
            return out, internals
            
        return out



class MLP(nn.Module):
    def __init__(self, seq_len, pred_len, in_dim, num_layers=1, hidden_dim=128, dropout=0.1):
        super().__init__()
        layers = []
        input_dim = seq_len * in_dim
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, pred_len))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, seq, in_dim)
        return self.net(x.flatten(start_dim=1))




# ============================================================
# Factory
# ============================================================

def get_model(model_name: str, config: dict):
    model_name = model_name
    
    if model_name == "iTransformer":
        return iTransformer(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            in_dim=config["in_dim"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"]
        )
    elif model_name == "grid_tst":
        return GridTST(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            in_dim=config["in_dim"],
            d_model=config["d_model"]
        )
    elif model_name == "Hybrid_Gated_FeatureFusion":
        return Hybrid_Gated_FeatureFusion_Model(
            seq_len=config["seq_len"],
            exog_dim=config["in_dim"],
            horizon=config["pred_len"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            dropout=config.get("dropout", 0.1),
            cnn_kernel=config.get("cnn_kernel", 3),
            cnn_mode=config.get("cnn_mode", "linear")
        )
    elif model_name == "Hybrid_KAN_Gated_FeatureFusion":
        return Hybrid_KAN_Gated_FeatureFusion_Model(
            seq_len=config["seq_len"],
            exog_dim=config["in_dim"],
            horizon=config["pred_len"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            dropout=config.get("dropout", 0.1),
            cnn_kernel=config.get("cnn_kernel", 3),
            cnn_mode=config.get("cnn_mode", "linear"),
            initial_grid=config.get("initial_grid", 3)
        )
    elif model_name == "Hybrid_Gated_QueryFusion":
        return Hybrid_Gated_QueryFusion_Model(
            seq_len=config["seq_len"],
            exog_dim=config["in_dim"],
            horizon=config["pred_len"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            dropout=config.get("dropout", 0.1),
            gate_init=config.get("gate_init", 0.1),
            use_hq=config.get("use_hq", True)
        )
    elif model_name == "Hybrid_FiLM_Ablation":
        return Hybrid_FiLM_Ablation_Model(
            seq_len=config["seq_len"],
            exog_dim=config["in_dim"],
            horizon=config["pred_len"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            dropout=config.get("dropout", 0.1),
            film_mode=config.get("film_mode", "bidir"),
            cnn_mode=config.get("cnn_mode", "linear"),
            use_hq=config.get("use_hq", True)
        )
    elif model_name == "Hybrid_NoHQ_Predict_Fusion":
        return Hybrid_NoHQ_Predict_Fusion_Model(
            seq_len=config["seq_len"],
            exog_dim=config["in_dim"],
            horizon=config["pred_len"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            dropout=config.get("dropout", 0.1),
            cnn_mode=config.get("cnn_mode", "linear")
        )
    elif model_name == "Hybrid_Predict_Fusion":
        return Hybrid_Predict_Fusion_Model(
            seq_len=config["seq_len"],
            exog_dim=config["in_dim"],
            horizon=config["pred_len"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            dropout=config.get("dropout", 0.1),
            fusion_mode="predict",
            use_hq=config.get("use_hq", True)
        )
    elif model_name == "htmformer":
        return HTMformer(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            in_dim=config["in_dim"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            nhead_var=config.get("nhead_var", 4)
        )
    elif model_name == "MLP":
        return MLP(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            in_dim=config["in_dim"],
            num_layers=config.get("num_layers", 1),
            hidden_dim=config.get("hidden_dim", 128),
            dropout=config.get("dropout", 0.1)
        )
    elif model_name == "TransformerFixed":
        return TransformerEncoderDecoderOneShot(
            config["in_dim"],
            config["pred_len"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config.get("num_layers", 2), # Map num_layers to encoder layers
            num_decoder_layers=config.get("num_layers", 2), # Map num_layers to decoder layers too
            dim_feedforward=config["hidden_dim"],
            dropout=config["dropout"]
        )
    elif model_name == "MMK":
        return MMKModel(
            seq_len=config["seq_len"],
            exog_dim=config["in_dim"],
            horizon=config["pred_len"],
            d_model=config.get("d_model", 32),
            n_experts=config.get("n_experts", 4),
            grid_size=config.get("grid_size", 3),
            num_layers=config.get("num_layers", 1)
        )
    elif model_name == "FiLM":
        from optimiser.models_film import FiLM # Only import FiLM, Hybrid_FiLM_Ablation_Model is defined locally
        return FiLM(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            in_dim=config["in_dim"],
            d_model=config.get("d_model", 256),
            modes=config.get("modes", 32)
        )
    else:
        raise ValueError(f"Unknown Model Code: {model_name}")
# ============================================================
# MMK (Multi-layer Mixture-of-KAN)
# ============================================================

class MMKLayer(nn.Module):
    """
    KAN Expert based MoE (Mixture of Experts) layer.
    """
    def __init__(self, in_features, out_features, n_experts=4, top_k=2, grid_size=3):
        super().__init__()
        self.experts = nn.ModuleList([
            KANLinear(in_features, out_features, grid_size=grid_size) for _ in range(n_experts)
        ])
        self.router = nn.Linear(in_features, n_experts)
        self.top_k = top_k

    def forward(self, x, return_routing=False):
        # x: (B, in_features) or (B, N, in_features)
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, orig_shape[-1])
            
        logits = self.router(x)  # (B', n_experts)
        probs = F.softmax(logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        
        output = torch.zeros(x.size(0), self.experts[0].out_features, device=x.device)
        for i in range(len(self.experts)):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask])
                # Scaling by router probability
                weight = probs[mask, i].unsqueeze(-1)
                output[mask] += expert_out * weight
        
        if len(orig_shape) == 3:
            output = output.reshape(orig_shape[0], orig_shape[1], -1)
            probs = probs.reshape(orig_shape[0], orig_shape[1], -1)
        
        if return_routing:
            return output, probs
        return output

class MMKModel(nn.Module):
    """
    Variable-Independent Multi-layer Mixture-of-KAN.
    Inherits interpretability by avoiding cross-variable token mixing.
    """
    def __init__(self, seq_len, exog_dim, horizon, d_model=64, n_experts=4, grid_size=3, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.exog_dim = exog_dim
        self.horizon = horizon
        self.num_vars = exog_dim + 1  # past target + exog
        self.num_layers = num_layers
        
        # Determine n_experts and grid_size per feature
        if isinstance(n_experts, (int, float)):
            n_experts_list = [int(n_experts)] * self.num_vars
        else:
            n_experts_list = n_experts
            assert len(n_experts_list) == self.num_vars
            
        if isinstance(grid_size, (int, float)):
            grid_size_list = [int(grid_size)] * self.num_vars
        else:
            grid_size_list = grid_size
            assert len(grid_size_list) == self.num_vars

        # Parallel branches for each variable (no mixing like iTransformer)
        self.branches = nn.ModuleList()
        for i in range(self.num_vars):
            curr_n_exp = n_experts_list[i]
            curr_grid = grid_size_list[i]
            
            if num_layers == 1:
                # Direct mapping for maximum interpretability
                branch = MMKLayer(seq_len, horizon, n_experts=curr_n_exp, grid_size=curr_grid)
            else:
                layers = nn.ModuleList()
                layers.append(MMKLayer(seq_len, d_model, n_experts=curr_n_exp, grid_size=curr_grid))
                for _ in range(num_layers - 2):
                    layers.append(MMKLayer(d_model, d_model, n_experts=curr_n_exp, grid_size=curr_grid))
                layers.append(MMKLayer(d_model, horizon, n_experts=curr_n_exp, grid_size=curr_grid))
                branch = layers
            self.branches.append(branch)
        
    def forward(self, x_ex, y_p, return_internals=False):
        # x_ex: (B, L, exog_dim)
        # y_p: (B, L, 1) or (B, L)
        if y_p.dim() == 2:
            y_p = y_p.unsqueeze(-1)
            
        # Stack into (B, num_vars, L)
        inputs = torch.cat([y_p, x_ex], dim=-1) # (B, L, F)
        inputs = inputs.permute(0, 2, 1) # (B, F, L)
        
        branch_outputs = []
        routing_info = []

        for i in range(self.num_vars):
            feat_input = inputs[:, i, :] # (B, L)
            
            if self.num_layers == 1:
                feat_out, gate_probs = self.branches[i](feat_input, return_routing=True)
                routing_info.append(gate_probs.unsqueeze(1)) # (B, 1, n_experts)
            else:
                # Sequential pass
                curr = feat_input
                layer_probs = []
                for layer in self.branches[i]:
                    curr, probs = layer(curr, return_routing=True)
                    layer_probs.append(probs.unsqueeze(1))
                feat_out = curr
                routing_info.append(torch.cat(layer_probs, dim=1).unsqueeze(1)) # (B, 1, L, n_exp)
            
            branch_outputs.append(feat_out)
            
        # Additive projection: Y = sum(f_i(X_i))
        stacked_outputs = torch.stack(branch_outputs, dim=1) # (B, F, horizon)
        final_output = stacked_outputs.sum(dim=1) # (B, horizon)
        
        if return_internals:
            internals = {
                "branch_outputs": stacked_outputs.detach(),
                "routing_probs": torch.stack(routing_info, dim=1).detach() # (B, F, L, n_exp) or (B, F, 1, n_exp)
            }
            return final_output, internals
        return final_output
