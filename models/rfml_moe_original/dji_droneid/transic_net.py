"""TranSIC-Net: Transformer Network for DJI DroneID OFDM Demodulation.

End-to-end transformer-based OFDM symbol demodulation for extracting
DJI DroneID information (serial number, GPS, drone type) from raw IQ signals.

Based on: "TranSIC-Net: An End-to-End Transformer Network for OFDM Symbol
Demodulation with Validation on DroneID Signals" (Sensors 2025).

DJI DroneID OFDM Parameters (LTE-based):
    - Sample rate: 15.36 MHz
    - Subcarrier spacing: 15 kHz
    - FFT size: 1024
    - Long CP: 80 samples
    - Short CP: 72 samples
    - Bandwidth: 10 MHz
    - Center freq: 2.4 GHz
    - 9 OFDM symbols per frame
    - ZC reference sequences on symbols 4 and 6
    - QPSK modulation on data symbols
    - 600 active subcarriers (of 1024)
    - Turbo coding + CRC16
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rfml.dji_droneid")

# ============================================================
# DJI DroneID Constants
# ============================================================

SAMPLE_RATE = 15.36e6
SUBCARRIER_SPACING = 15e3
FFT_SIZE = 1024
LONG_CP_LEN = 80   # 15.36e6 / 192e3
SHORT_CP_LEN = 72  # 0.0000046875 * 15.36e6
NUM_OFDM_SYMBOLS = 9
ZC_SYMBOL_INDICES = [3, 5]  # 0-indexed: symbols 4 and 6 are ZC pilots
DATA_SYMBOL_INDICES = [0, 1, 2, 4, 6, 7, 8]  # 7 data symbols
NUM_DATA_CARRIERS = 600  # active subcarriers
BITS_PER_SYMBOL = 2  # QPSK

# Symbol lengths in samples
SYMBOL_WITH_LONG_CP = FFT_SIZE + LONG_CP_LEN   # 1104
SYMBOL_WITH_SHORT_CP = FFT_SIZE + SHORT_CP_LEN  # 1096

# Total frame length: 1 long CP + 8 short CP = 1104 + 8*1096 = 9872 samples
FRAME_LENGTH = SYMBOL_WITH_LONG_CP + (NUM_OFDM_SYMBOLS - 1) * SYMBOL_WITH_SHORT_CP

# DJI drone type lookup
DRONE_TYPES = {
    1: "Inspire 1", 2: "Phantom 3 Series", 4: "Phantom 3 Std",
    5: "M100", 11: "Phantom 4", 15: "Mavic Pro", 16: "Inspire 2",
    17: "Phantom 4 Pro", 20: "Spark", 23: "Mavic Air", 40: "Mavic 2",
    41: "Mavic 2 Pro", 51: "Mavic 2 Enterprise", 53: "Mavic Mini",
    58: "Mavic Air 2", 60: "M300 RTK", 61: "DJI FPV", 63: "Mini 2",
    66: "Air 2S", 68: "DJI Mavic 3", 70: "Mini SE", 92: "Mini 4",
    93: "Mini 4 Pro", 108: "Mini 4K",
}


# ============================================================
# Zadoff-Chu Sequence Generation
# ============================================================

def create_zc_sequence(root: int = 600, length: int = 601) -> torch.Tensor:
    """Generate a Zadoff-Chu sequence for channel estimation.

    Args:
        root: ZC sequence root index.
        length: Sequence length (prime number).

    Returns:
        Complex tensor of shape (length,).
    """
    n = torch.arange(length, dtype=torch.float64)
    phase = -math.pi * root * n * (n + 1) / length
    return torch.complex(torch.cos(phase), torch.sin(phase)).to(torch.complex64)


# ============================================================
# OFDM Signal Processing
# ============================================================

class OFDMFrameExtractor(nn.Module):
    """Extract and preprocess OFDM symbols from raw IQ for the transformer.

    Takes raw IQ data (already burst-extracted and resampled to 15.36 MHz),
    removes cyclic prefixes, applies FFT, and extracts data subcarriers.

    Input: [B, 2, frame_length] raw IQ (I and Q channels)
    Output: [B, num_data_symbols, num_data_carriers, 2] real-valued OFDM grid
    """

    def __init__(self, fft_size: int = FFT_SIZE, long_cp: int = LONG_CP_LEN,
                 short_cp: int = SHORT_CP_LEN, num_symbols: int = NUM_OFDM_SYMBOLS,
                 num_data_carriers: int = NUM_DATA_CARRIERS):
        super().__init__()
        self.fft_size = fft_size
        self.long_cp = long_cp
        self.short_cp = short_cp
        self.num_symbols = num_symbols
        self.num_data_carriers = num_data_carriers

        # Precompute data carrier indices (center 600 of 1024)
        # Active carriers: indices 212..811 (600 carriers centered in FFT)
        start = (fft_size - num_data_carriers) // 2
        self.register_buffer(
            "data_carrier_indices",
            torch.arange(start, start + num_data_carriers, dtype=torch.long),
        )

    def forward(self, iq: torch.Tensor) -> dict:
        """Extract OFDM symbols from raw IQ frame.

        Args:
            iq: [B, 2, N] raw IQ frame (I, Q channels).

        Returns:
            dict with:
                "freq_domain": [B, num_symbols, num_data_carriers, 2] — real/imag per carrier
                "zc_symbols": [B, 2, fft_size, 2] — ZC reference symbols for channel estimation
                "data_symbols": [B, 7, num_data_carriers, 2] — data symbols only
        """
        B = iq.shape[0]
        z = torch.complex(iq[:, 0, :], iq[:, 1, :])  # [B, N]

        # Extract each OFDM symbol by removing CP
        symbols_freq = []
        offset = 0
        for sym_idx in range(self.num_symbols):
            cp_len = self.long_cp if sym_idx == 0 else self.short_cp
            # Skip CP, take FFT_SIZE samples
            start = offset + cp_len
            end = start + self.fft_size
            if end > z.shape[1]:
                # Pad if needed
                sym = F.pad(z[:, start:], (0, end - z.shape[1]))
            else:
                sym = z[:, start:end]  # [B, fft_size]

            # FFT
            sym_freq = torch.fft.fft(sym, n=self.fft_size)  # [B, fft_size]
            sym_freq = torch.fft.fftshift(sym_freq, dim=-1)  # center DC
            symbols_freq.append(sym_freq)

            offset = end

        symbols_freq = torch.stack(symbols_freq, dim=1)  # [B, 9, fft_size]

        # Extract data carriers
        data_carriers = symbols_freq[:, :, self.data_carrier_indices]  # [B, 9, 600]

        # Separate ZC and data symbols
        zc_syms = symbols_freq[:, ZC_SYMBOL_INDICES, :]  # [B, 2, fft_size]
        data_syms = data_carriers[:, DATA_SYMBOL_INDICES, :]  # [B, 7, 600]

        # Convert to real-valued [real, imag] for transformer input
        def to_real_pair(x):
            return torch.stack([x.real, x.imag], dim=-1)  # [..., 2]

        return {
            "freq_domain": to_real_pair(data_carriers),  # [B, 9, 600, 2]
            "zc_symbols": to_real_pair(zc_syms),  # [B, 2, fft_size, 2]
            "data_symbols": to_real_pair(data_syms),  # [B, 7, 600, 2]
            "all_symbols_complex": symbols_freq,  # [B, 9, fft_size] complex
        }


# ============================================================
# Channel Estimation
# ============================================================

class ZCChannelEstimator(nn.Module):
    """Estimate channel response from ZC reference symbols.

    Uses the known ZC sequences on OFDM symbols 4 and 6 to estimate
    the frequency-domain channel response H(f), then interpolates
    across all subcarriers and symbols.

    Args:
        fft_size: FFT size (1024).
        num_data_carriers: Active subcarriers (600).
        zc_root: Zadoff-Chu root index.
    """

    def __init__(self, fft_size: int = FFT_SIZE,
                 num_data_carriers: int = NUM_DATA_CARRIERS,
                 zc_root: int = 600):
        super().__init__()
        self.fft_size = fft_size
        self.num_data_carriers = num_data_carriers

        # Generate reference ZC sequence in frequency domain
        zc = create_zc_sequence(zc_root, num_data_carriers)
        zc_full = torch.zeros(fft_size, dtype=torch.complex64)
        start = (fft_size - num_data_carriers) // 2
        zc_full[start:start + num_data_carriers] = zc[:num_data_carriers]
        self.register_buffer("zc_ref", zc_full)  # [fft_size]

    def forward(self, zc_received: torch.Tensor) -> torch.Tensor:
        """Estimate channel from received ZC symbols.

        Args:
            zc_received: [B, 2, fft_size] complex received ZC symbols.

        Returns:
            [B, num_data_carriers] complex channel estimates (averaged over 2 ZC symbols).
        """
        # LS channel estimation: H = Y / X (element-wise division)
        zc_ref = self.zc_ref.unsqueeze(0)  # [1, fft_size]
        ref_nonzero = zc_ref.abs() > 1e-10

        h_estimates = []
        for i in range(2):  # two ZC symbols
            y = zc_received[:, i, :]  # [B, fft_size]
            h = torch.where(ref_nonzero, y / (zc_ref + 1e-10), torch.zeros_like(y))
            h_estimates.append(h)

        # Average the two estimates
        h_avg = (h_estimates[0] + h_estimates[1]) / 2  # [B, fft_size]

        # Extract data carrier channel estimates
        start = (self.fft_size - self.num_data_carriers) // 2
        h_data = h_avg[:, start:start + self.num_data_carriers]  # [B, 600]

        return h_data


# ============================================================
# TranSIC-Net: Transformer for OFDM Demodulation
# ============================================================

class SubcarrierEmbedding(nn.Module):
    """Embed OFDM subcarrier values (real+imag) into transformer tokens.

    Each subcarrier's [real, imag] pair is projected to d_model dimensions.
    Positional encoding indicates the subcarrier index within the OFDM symbol.

    Args:
        d_model: Transformer model dimension.
        num_carriers: Number of data subcarriers.
        max_symbols: Maximum number of OFDM symbols.
    """

    def __init__(self, d_model: int = 128, num_carriers: int = NUM_DATA_CARRIERS,
                 max_symbols: int = NUM_OFDM_SYMBOLS):
        super().__init__()
        self.d_model = d_model

        # Project [real, imag] -> d_model
        self.value_proj = nn.Linear(2, d_model)

        # Learnable subcarrier positional encoding
        self.carrier_pos = nn.Parameter(torch.randn(1, num_carriers, d_model) * 0.02)

        # Learnable symbol index embedding
        self.symbol_embed = nn.Embedding(max_symbols, d_model)

    def forward(self, x: torch.Tensor, symbol_idx: int = 0) -> torch.Tensor:
        """Embed one OFDM symbol's subcarriers.

        Args:
            x: [B, num_carriers, 2] real/imag values.
            symbol_idx: Which OFDM symbol (0-8) for positional embedding.

        Returns:
            [B, num_carriers, d_model] embedded tokens.
        """
        tokens = self.value_proj(x)  # [B, num_carriers, d_model]
        tokens = tokens + self.carrier_pos  # add subcarrier position
        sym_emb = self.symbol_embed(torch.tensor(symbol_idx, device=x.device))
        tokens = tokens + sym_emb.unsqueeze(0).unsqueeze(0)  # add symbol position
        return tokens


class SICBlock(nn.Module):
    """Successive Interference Cancellation block.

    After initial demodulation of each symbol, the SIC block re-encodes
    the estimated symbols and subtracts their interference from subsequent
    symbols, improving demodulation accuracy iteratively.

    Args:
        d_model: Model dimension.
        num_carriers: Number of data subcarriers.
    """

    def __init__(self, d_model: int = 128, num_carriers: int = NUM_DATA_CARRIERS):
        super().__init__()
        # Re-encode estimated bits back to constellation points
        self.bit_to_symbol = nn.Linear(2, 2)  # soft bits -> soft constellation
        # Interference estimation
        self.interference_est = nn.Sequential(
            nn.Linear(d_model + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),  # estimated interference [real, imag]
        )

    def forward(self, received: torch.Tensor, estimated_bits: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        """Cancel estimated interference from received signal.

        Args:
            received: [B, num_carriers, 2] received signal (real/imag).
            estimated_bits: [B, num_carriers, 2] soft bit estimates from first pass.
            features: [B, num_carriers, d_model] transformer features.

        Returns:
            [B, num_carriers, 2] interference-cancelled signal.
        """
        # Estimate the interference contribution
        combined = torch.cat([features, estimated_bits], dim=-1)
        interference = self.interference_est(combined)

        # Subtract estimated interference
        cleaned = received - interference
        return cleaned


class TranSICNet(nn.Module):
    """TranSIC-Net: End-to-End Transformer for OFDM Symbol Demodulation.

    Processes received OFDM symbols through:
    1. Subcarrier embedding with positional encoding
    2. Transformer encoder for inter-subcarrier attention
    3. QPSK bit prediction per subcarrier
    4. Optional SIC refinement pass

    The model takes frequency-domain OFDM symbols (after FFT + CP removal)
    and outputs soft bit estimates for each data subcarrier.

    Args:
        d_model: Transformer dimension (default 128).
        num_heads: Number of attention heads (default 4).
        num_layers: Transformer encoder layers (default 4).
        d_ff: Feed-forward dimension (default 512).
        num_carriers: Data subcarriers (default 600).
        num_data_symbols: Data OFDM symbols per frame (default 7).
        dropout: Dropout rate (default 0.1).
        use_sic: Enable SIC refinement (default True).
        sic_iterations: Number of SIC passes (default 2).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        num_carriers: int = NUM_DATA_CARRIERS,
        num_data_symbols: int = 7,
        dropout: float = 0.1,
        use_sic: bool = True,
        sic_iterations: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_carriers = num_carriers
        self.num_data_symbols = num_data_symbols
        self.use_sic = use_sic
        self.sic_iterations = sic_iterations

        # Subcarrier embedding
        self.embedding = SubcarrierEmbedding(d_model, num_carriers)

        # Channel condition embedding (optional: feed channel estimate)
        self.channel_proj = nn.Linear(2, d_model)  # channel H [real, imag] -> d_model

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # QPSK bit predictor: predicts 2 soft bits per subcarrier
        self.bit_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, BITS_PER_SYMBOL),  # 2 bits for QPSK
        )

        # SIC block for iterative refinement
        if use_sic:
            self.sic_block = SICBlock(d_model, num_carriers)
            # Second-pass encoder (lighter)
            sic_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff // 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sic_encoder = nn.TransformerEncoder(sic_encoder_layer, num_layers=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def demodulate_symbol(
        self,
        symbol_data: torch.Tensor,
        channel_h: Optional[torch.Tensor] = None,
        symbol_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Demodulate a single OFDM symbol.

        Args:
            symbol_data: [B, num_carriers, 2] received signal (real/imag).
            channel_h: [B, num_carriers] complex channel estimate (optional).
            symbol_idx: OFDM symbol index for positional embedding.

        Returns:
            bits: [B, num_carriers, 2] soft bit predictions.
            features: [B, num_carriers, d_model] transformer features.
        """
        # Embed subcarriers
        tokens = self.embedding(symbol_data, symbol_idx)  # [B, carriers, d_model]

        # Optionally add channel information
        if channel_h is not None:
            h_real = torch.stack([channel_h.real, channel_h.imag], dim=-1)
            h_emb = self.channel_proj(h_real)  # [B, carriers, d_model]
            tokens = tokens + h_emb

        # Transformer encoding
        features = self.encoder(tokens)  # [B, carriers, d_model]
        features = self.encoder_norm(features)

        # Predict bits
        bits = self.bit_predictor(features)  # [B, carriers, 2]

        return bits, features

    def forward(
        self,
        data_symbols: torch.Tensor,
        channel_h: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass: demodulate all data OFDM symbols in a frame.

        Args:
            data_symbols: [B, num_data_symbols, num_carriers, 2] real/imag data.
            channel_h: [B, num_carriers] complex channel estimate (optional).

        Returns:
            dict with:
                "bits": [B, num_data_symbols, num_carriers, 2] soft bit estimates
                "bits_hard": [B, num_data_symbols, num_carriers, 2] hard decisions (0/1)
                "bits_flat": [B, total_bits] flattened hard bit sequence
        """
        B = data_symbols.shape[0]
        all_bits = []
        all_features = []

        # First pass: demodulate each symbol independently
        for sym_idx in range(self.num_data_symbols):
            sym_data = data_symbols[:, sym_idx, :, :]  # [B, carriers, 2]
            bits, features = self.demodulate_symbol(sym_data, channel_h, sym_idx)
            all_bits.append(bits)
            all_features.append(features)

        bits_pass1 = torch.stack(all_bits, dim=1)  # [B, 7, carriers, 2]

        # SIC refinement pass
        if self.use_sic:
            refined_bits = []
            for sic_iter in range(self.sic_iterations):
                for sym_idx in range(self.num_data_symbols):
                    sym_data = data_symbols[:, sym_idx, :, :]
                    est_bits = all_bits[sym_idx]
                    feat = all_features[sym_idx]

                    # Cancel interference
                    cleaned = self.sic_block(sym_data, est_bits.detach(), feat.detach())

                    # Re-embed and re-encode
                    tokens = self.embedding(cleaned, sym_idx)
                    if channel_h is not None:
                        h_real = torch.stack([channel_h.real, channel_h.imag], dim=-1)
                        tokens = tokens + self.channel_proj(h_real)

                    refined_feat = self.sic_encoder(tokens)
                    refined = self.bit_predictor(refined_feat)
                    all_bits[sym_idx] = refined
                    all_features[sym_idx] = refined_feat

            bits_final = torch.stack(all_bits, dim=1)
        else:
            bits_final = bits_pass1

        # Hard decisions
        bits_hard = (torch.sigmoid(bits_final) > 0.5).float()

        # Flatten to bit sequence
        bits_flat = bits_hard.reshape(B, -1)  # [B, 7*600*2 = 8400 bits]

        return {
            "bits": bits_final,           # soft bits [B, 7, 600, 2]
            "bits_hard": bits_hard,       # hard bits [B, 7, 600, 2]
            "bits_flat": bits_flat,       # [B, 8400]
            "bits_pass1": bits_pass1,     # first pass (before SIC)
        }

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Full DroneID Demodulation Pipeline
# ============================================================

class DroneIDDemodulator(nn.Module):
    """Complete DJI DroneID demodulation pipeline.

    Combines:
    1. OFDM frame extraction (CP removal + FFT)
    2. ZC-based channel estimation
    3. TranSIC-Net transformer demodulation
    4. Bit descrambling and decoding

    Input: Raw IQ burst [B, 2, N] at 15.36 MHz
    Output: Demodulated bit sequence and decoded DroneID fields

    Args:
        d_model: Transformer dimension.
        num_heads: Attention heads.
        num_layers: Encoder layers.
        use_sic: Enable SIC refinement.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        use_sic: bool = True,
    ):
        super().__init__()
        self.frame_extractor = OFDMFrameExtractor()
        self.channel_estimator = ZCChannelEstimator()
        self.demodulator = TranSICNet(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            use_sic=use_sic,
        )

    def forward(self, iq: torch.Tensor) -> dict:
        """Full demodulation pipeline.

        Args:
            iq: [B, 2, N] raw IQ frame at 15.36 MHz sample rate.

        Returns:
            dict with demodulated bits and intermediate results.
        """
        # Step 1: Extract OFDM symbols
        frame = self.frame_extractor(iq)

        # Step 2: Channel estimation from ZC symbols
        zc_complex = frame["all_symbols_complex"][:, ZC_SYMBOL_INDICES, :]
        channel_h = self.channel_estimator(zc_complex)

        # Step 3: Transformer demodulation
        result = self.demodulator(frame["data_symbols"], channel_h)

        # Add frame info to result
        result["channel_h"] = channel_h
        result["frame"] = frame

        return result

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Training Utilities
# ============================================================

class QPSKLoss(nn.Module):
    """Combined loss for QPSK demodulation training.

    Binary cross-entropy on soft bit predictions, with optional
    constellation MSE regularization.

    Args:
        bce_weight: Weight for BCE loss (default 1.0).
        mse_weight: Weight for constellation MSE (default 0.1).
    """

    def __init__(self, bce_weight: float = 1.0, mse_weight: float = 0.1):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight

    def forward(self, predictions: dict, target_bits: torch.Tensor) -> dict:
        """Compute demodulation loss.

        Args:
            predictions: dict from TranSICNet.forward().
            target_bits: [B, num_symbols, num_carriers, 2] ground truth bits.

        Returns:
            dict with "total", "bce", "mse" losses.
        """
        soft_bits = predictions["bits"]

        # BCE loss on soft bit estimates
        bce = F.binary_cross_entropy_with_logits(soft_bits, target_bits)

        # MSE on first-pass bits (encourage early convergence)
        mse = F.mse_loss(torch.sigmoid(predictions["bits_pass1"]), target_bits)

        total = self.bce_weight * bce + self.mse_weight * mse

        return {"total": total, "bce": bce, "mse": mse}
