import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LegendreProjection(nn.Module):
    def __init__(self, order, input_len, device='cpu', dtype=torch.float32):
        super(LegendreProjection, self).__init__()
        self.order = order
        self.input_len = input_len
        self.device = device
        self.dtype = dtype
        
        # Initialize Legendre Polynomials matrix
        # Domain [-1, 1]
        t = torch.linspace(-1, 1, input_len).to(device)
        self.register_buffer('t', t)
        
        # P matrix: (order, input_len)
        P = self._generate_legendre(order, t)
        self.register_buffer('P', P)
        
    def _generate_legendre(self, order, t):
        # Generates Legendre polynomials up to 'order'
        # shape: (order, len(t))
        P = torch.zeros(order, len(t), device=t.device, dtype=self.dtype)
        P[0] = 1.0
        if order > 1:
            P[1] = t
        for n in range(1, order - 1):
            # Recurrence relation: (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
            # P_{n+1} = ((2n+1)xP_n - nP_{n-1}) / (n+1)
            P[n+1] = ((2*n + 1) * t * P[n] - n * P[n-1]) / (n + 1)
        return P

    def forward(self, x):
        # x: (B, D, L)
        # Project to Legendre coefficients: C = X @ P.T (checking shapes)
        # X: (B, D, L), P: (K, L) -> C: (B, D, K)
        # We want to project the time dimension L onto K bases.
        # inner product: integral(f(t)P(t)dt). Discretized as sum.
        # We can use a simple matrix multiplication.
        # Adjusted for orthogonality normalization if needed, but standard projection is fine for learned weights often.
        P = self.P # (K, L)
        
        # (B, D, L) @ (L, K) -> (B, D, K)
        # transpose P to (L, K)
        return torch.matmul(x, P.T) 

    def reconstruct(self, c, pred_len):
        # Reconstruct from coefficients
        # Needs P matrix for the target horizon?
        # Usually FiLM projects back to new horizon.
        # We need P_out for t_out in domain? 
        # Typically time-series forecasting maps Input Domain -> Output Domain.
        # If we assume we predict coefficients C', then reconstruct C' @ P_out.
        # But standard FiLM might map C -> C' (via FEL) -> reconstruct.
        
        # Let's generate P for prediction length. We align domain [-1, 1] to the output horizon?
        # Or usually it projects back to the same domain length and then slices?
        # Looking at paper: They often reconstruct for the *prediction horizon*.
        # Let's assume we map to t in [-1, 1] for the horizon length.
        t_out = torch.linspace(-1, 1, pred_len, device=self.P.device)
        P_out = self._generate_legendre(self.order, t_out) # (K, pred_len)
        
        # c: (B, D, K)
        # out: c @ P_out -> (B, D, K) @ (K, pred_len) -> (B, D, pred_len)
        return torch.matmul(c, P_out)


class FrequencyEnhancedBlock(nn.Module):
    def __init__(self, in_channels, seq_len, modes=32, mode_select_method='random'):
        super(FrequencyEnhancedBlock, self).__init__()
        self.modes = modes
        self.mode_select_method = mode_select_method
        
        # Weights for frequency combination
        # Input is (B, D, K). We treat K (order) as the sequence length for FFT?
        # Actually in FiLM, FEL is applied on the Legendre Coefficients C? 
        # Or usually FEL is widely used on the time domain.
        # The paper says: "Legendre Projection Unit (LPU) ... and Frequency Enhanced Layer (FEL)."
        # "LPU ... projects historical series into Legendre coefficients."
        # "FEL ... is applied to the coefficients."
        # So input to this block is (B, D, K).
        
        # We use a simplified version of Fourier interaction (like Autoformer/FEDformer ideas)
        # but customized.
        # Let's assume we learn a weight matrix to mix frequencies.
        
        # We need a defined scale. 
        # Let's implement a learnable spectral filter.
        self.scale = 0.02
        self.w = nn.Parameter(self.scale * torch.randn(in_channels, in_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        # x: (B, D, K) -> Coefficients
        B, D, K = x.shape
        
        # FFT on the coefficient dimension K? 
        # Or is x the original time series?
        # In FiLM, LPU -> FEL. So x is coefficients.
        # K is usually small (e.g. 64 or 256).
        
        x_ft = torch.fft.rfft(x, dim=-1) # (B, D, K//2 + 1)
        
        # Select modes
        # We use 'modes' to limit the frequency range processed.
        train_modes = min(self.modes, x_ft.shape[-1])
        
        # Apply weights
        # (B, D, modes)
        out_ft = torch.zeros_like(x_ft)
        
        # Mixing channels? "Channel Independence" usually means we don't mix D.
        # BUT the definition of FEL often involves mixing or just per-channel weight.
        # If we stick to strict Channel Independence, we should broadcast or use per-channel weights.
        # The weight self.w is (D, D, modes) which implies mixing D.
        # If we want CI, we should use (1, 1, modes) or (D, 1, modes) (diagonal).
        # Let's use Channel Independent weight: (1, 1, modes) or just elementwise.
        # To be safe and efficient for CI:
        # We will use distinct weights for each mode but shared across channels if purely CI, 
        # or just learn a complex weight vector.
        
        # Let's update self.w to be consistent with Channel Independence (shared weights for all channels often better for generalization).
        # Or independent weights per channel but no mixing.
        # Given "iTransformer" comparison, CI is key.
        # We'll use: weight (modes,) complex.
        # Wait, if we define self.w as (D, D, modes) above, that's mixing.
        pass # Re-defined in init below
        
        return x # Placeholder for the class structure

class FrequencyEnhancedBlockCI(nn.Module):
    """Channel Independent Frequency Enhanced Block"""
    def __init__(self, modes=32):
        super().__init__()
        self.modes = modes
        # Weight: (modes,)
        self.scale = 0.02
        self.w = nn.Parameter(self.scale * torch.randn(modes, dtype=torch.cfloat))

    def forward(self, x):
        # x: (B, D, K)
        B, D, K = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Apply interaction
        # We just multiply by learnable weights in frequency domain
        train_modes = min(self.modes, x_ft.shape[-1])
        
        # Operation: element-wise mult for top modes
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :train_modes] = x_ft[:, :, :train_modes] * self.w[:train_modes]
        
        # Inverse FFT
        x_out = torch.fft.irfft(out_ft, n=K, dim=-1)
        return x_out


class FiLM(nn.Module):
    """
    FiLM: Frequency improved Legendre Memory Model
    """
    def __init__(self, seq_len, pred_len, in_dim, 
                 d_model=256, # Represents Legendre Order K here? Or hidden dim?
                 # In FiLM paper, they map seq_len -> N Legendre bases.
                 # Let's treat d_model as the 'Legendre Order' (K).
                 modes=32,
                 **kwargs):
        super(FiLM, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.legendre_order = d_model # Use d_model argument as the order K
        self.modes = modes
        
        # Legendre Projection Unit
        self.lpu = LegendreProjection(self.legendre_order, seq_len)
        
        # Frequency Enhanced Layer
        # We assume one FEL block for simplicity, or stack them.
        # Paper uses 1 layer usually sufficient.
        self.fel = FrequencyEnhancedBlockCI(modes=modes)
        
        # Final projection to prediction horizon happens in reconstruction
        # But we might need a linear mapping if we want to change characteristics.
        # FiLM paper: C -> FEL(C) -> Project(C) -> Reconstruct.
        # Or C -> FEL(C) -> Reconstruct(C) -> Time Domain.
        # If we reconstruct to 'pred_len', we rely on P_out.
        
        # Optional: A linear layer on coefficients to allow mapping historical structure to future structure
        self.map_layer = nn.Linear(self.legendre_order, self.legendre_order)
        
    def forward(self, x, return_internals=False):
        # x: (B, L, D) - Multivariate input
        B, L, D = x.shape
        
        # 1. Normalization (RevIN equivalent - simplified)
        # Just simple standardization per instance
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - mean) / std
        
        # 2. Permute for Channel Independence processing
        # We want to treat D as Batch: (B*D, L, 1) or just operate on (B, D, L)
        x_perm = x_norm.permute(0, 2, 1) # (B, D, L)
        
        # 3. Legendre Projection
        # (B, D, L) -> (B, D, K)
        coeffs = self.lpu(x_perm)
        
        # 4. Frequency Enhanced Layer
        # (B, D, K) -> (B, D, K)
        coeffs_enh = self.fel(coeffs)
        
        # 5. Mapping (Optional but good for forecasting)
        # Learn transition for coefficients
        coeffs_map = self.map_layer(coeffs_enh)
        
        # 6. Legendre Reconstruction
        # (B, D, K) -> (B, D, H)
        # Reconstructs signal for 'pred_len'
        out_perm = self.lpu.reconstruct(coeffs_map, self.pred_len)
        
        # 7. Denormalization & Permute back
        # out_perm: (B, D, H) -> (B, H, D)
        output = out_perm.permute(0, 2, 1)
        
        # Denorm (using historical stats, assuming stationarity locally)
        output = output * std + mean
        
        if return_internals:
            return output, {"coeffs": coeffs, "coeffs_enh": coeffs_enh}
            
        return output
