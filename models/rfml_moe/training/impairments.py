import math
import torch
import torch.nn as nn

class CarrierFrequencyOffset(nn.Module):
    """Applies random Carrier Frequency Offset (CFO) to IQ signals.
    Formula: r_CFO[n] = r[n] * e^{j(2\pi \Delta f/f_s n + \theta)}
    
    Operates on batch tensors of shape (B, 2, N).
    """
    def __init__(self, max_offset: float = 0.001):
        super().__init__()
        self.max_offset = max_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, N)
        B, _, N = x.shape
        device = x.device
        
        # Random frequency offset per sample in batch
        df = (torch.rand(B, 1, device=device) * 2 - 1) * self.max_offset
        # Random initial phase
        theta = torch.rand(B, 1, device=device) * 2 * math.pi
        
        # Time index vector [0, 1, ..., N-1]
        n = torch.arange(N, device=device).view(1, -1)
        
        # Compute phase: 2 * pi * df * n + theta
        phase = 2 * math.pi * df * n + theta
        
        cos_p = torch.cos(phase).unsqueeze(1) # (B, 1, N)
        sin_p = torch.sin(phase).unsqueeze(1) # (B, 1, N)
        
        i, q = x[:, 0:1, :], x[:, 1:2, :]
        
        # Complex multiplication: (i + jq) * (cos + jsin)
        # i_new = i*cos - q*sin
        # q_new = i*sin + q*cos
        i_new = i * cos_p - q * sin_p
        q_new = i * sin_p + q * cos_p
        
        return torch.cat([i_new, q_new], dim=1)

class IQImbalance(nn.Module):
    """Applies IQ Imbalance (amplitude and phase mismatch) to signals.
    Formula: r_IQI = \mu * r + \nu * r^*
    where:
    \mu = cos(phi/2) + j * g * sin(phi/2)
    \nu = g * cos(phi/2) - j * sin(phi/2)
    """
    def __init__(self, max_gain_db: float = 1.0, max_phase_deg: float = 5.0):
        super().__init__()
        self.max_gain_db = max_gain_db
        self.max_phase_deg = max_phase_deg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, N)
        B, _, N = x.shape
        device = x.device
        
        # Random gain and phase imbalance
        g_db = (torch.rand(B, 1, 1, device=device) * 2 - 1) * self.max_gain_db
        g = 10 ** (g_db / 20)
        phi = ((torch.rand(B, 1, 1, device=device) * 2 - 1) * self.max_phase_deg) * math.pi / 180
        
        cos_p = torch.cos(phi/2)
        sin_p = torch.sin(phi/2)
        
        # Mu and Nu components
        # mu = cos_p + j * g * sin_p
        # nu = g * cos_p - j * sin_p
        
        i, q = x[:, 0:1, :], x[:, 1:2, :]
        
        # r_iqi = mu * (i + jq) + nu * (i - jq)
        #       = (mu_real + j mu_imag) * (i + jq) + (nu_real + j nu_imag) * (i - jq)
        # i_new = mu_r*i - mu_i*q + nu_r*i + nu_i*q
        # q_new = mu_r*q + mu_i*i - nu_r*q + nu_i*i
        
        mu_r, mu_i = cos_p, g * sin_p
        nu_r, nu_i = g * cos_p, -sin_p
        
        i_new = mu_r * i - mu_i * q + nu_r * i + nu_i * q
        q_new = mu_r * q + mu_i * i - nu_r * q + nu_i * i
        
        return torch.cat([i_new, q_new], dim=1)

class SignalAugmentationPipeline(nn.Module):
    """Sim-to-Real augmentation pipeline for RF signals."""
    def __init__(self, cfo_limit: float, iq_gain_limit: float, iq_phase_limit: float):
        super().__init__()
        self.cfo = CarrierFrequencyOffset(max_offset=cfo_limit)
        self.iqi = IQImbalance(max_gain_db=iq_gain_limit, max_phase_deg=iq_phase_limit)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cfo(x)
        x = self.iqi(x)
        return x
