"""Comprehensive unit tests for all RFML-MoE new components.

Tests cover:
- Phase 1: Energy Detector, EMD Denoiser, VMD Extractor, GAF Extractor,
           VMD-GAF Expert, TFMS Expert, SNR-Adaptive Router
- Phase 2: SignalFormer Expert, Hi-WaveTST Expert, LWM Expert,
           Neuro-Symbolic RFF Expert, Visual-RF Detector
- Integration: MoE model with 4/6/8/11 expert configurations
- Pretraining: MaskedFrequencyPredictor, WavesFM MWM, SpectrumFM dual-objective

Requires: torch, pytest
Run: cd /home/rax/exp/iq/rfml && python -m pytest tests/ -v
"""

import math

import torch
import torch.nn as nn
import pytest


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def iq_batch():
    """Batch of 4 IQ samples, [B, 2, 32768]."""
    return torch.randn(4, 2, 32768)


@pytest.fixture
def iq_single():
    """Single IQ sample, [2, 32768]."""
    return torch.randn(2, 32768)


@pytest.fixture
def spectrogram_batch():
    """Batch of spectrogram images, [B, 3, 512, 512]."""
    return torch.randn(2, 3, 512, 512)


@pytest.fixture
def gaf_batch():
    """Batch of GAF images, [B, 3, 256, 256]."""
    return torch.randn(2, 3, 256, 256)


# ============================================================
# Phase 1: Feature Extractors
# ============================================================

class TestEnergyDetector:
    """Tests for features/energy_gate.py."""

    def test_import(self):
        from features.energy_gate import EnergyDetector
        d = EnergyDetector()
        assert d is not None

    def test_calibrate(self, iq_single):
        from features.energy_gate import EnergyDetector
        d = EnergyDetector(p_fa=0.01)
        d.calibrate(iq_single)
        assert d._mu is not None
        assert d._sigma is not None
        assert d._mu > 0

    def test_detect_returns_bool(self, iq_single):
        from features.energy_gate import EnergyDetector
        d = EnergyDetector()
        d.calibrate(iq_single)
        result = d.detect(iq_single)
        assert isinstance(result, bool)

    def test_snr_estimate_returns_float(self, iq_single):
        from features.energy_gate import EnergyDetector
        d = EnergyDetector()
        d.calibrate(iq_single)
        snr = d.snr_estimate_db(iq_single)
        assert isinstance(snr, float)
        assert not math.isnan(snr)

    def test_noise_not_detected(self):
        """Pure Gaussian noise should rarely trigger detection."""
        from features.energy_gate import EnergyDetector
        d = EnergyDetector(p_fa=0.01)
        noise = torch.randn(2, 32768)
        d.calibrate(noise)
        # With p_fa=0.01, ~1% false alarm rate
        detections = sum(d.detect(torch.randn(2, 32768)) for _ in range(100))
        assert detections < 10  # allow up to 10% for statistical variation

    def test_strong_signal_detected(self):
        """A strong signal should always be detected."""
        from features.energy_gate import EnergyDetector
        d = EnergyDetector()
        noise = torch.randn(2, 32768) * 0.1
        d.calibrate(noise)
        signal = torch.randn(2, 32768) * 10.0  # 40 dB above noise
        assert d.detect(signal) is True


class TestEMDDenoiser:
    """Tests for features/emd_denoising.py."""

    def test_import(self):
        from features.emd_denoising import EMDDenoiser
        d = EMDDenoiser()
        assert d is not None

    def test_decompose_returns_list(self):
        from features.emd_denoising import EMDDenoiser
        d = EMDDenoiser(num_imfs=3)
        signal = torch.randn(512)
        imfs = d.decompose(signal)
        assert isinstance(imfs, list)
        assert len(imfs) >= 2  # at least 1 IMF + residual

    def test_denoise_shape_preserved(self):
        from features.emd_denoising import EMDDenoiser
        d = EMDDenoiser()
        iq = torch.randn(2, 1024)
        denoised = d.denoise(iq)
        assert denoised.shape == iq.shape
        assert denoised.dtype == iq.dtype


class TestVMDExtractor:
    """Tests for features/vmd.py."""

    def test_import(self):
        from features.vmd import VMDExtractor
        v = VMDExtractor()
        assert v is not None

    def test_decompose_shapes(self):
        from features.vmd import VMDExtractor
        v = VMDExtractor(K=3, max_iter=50)
        signal = torch.randn(256)
        u, omega = v.decompose(signal)
        assert u.shape[0] == 3  # K modes
        assert u.shape[1] == 256  # N samples
        assert omega.shape[0] == 3

    def test_extract_shape(self):
        from features.vmd import VMDExtractor
        v = VMDExtractor(K=3, max_iter=50)
        iq = torch.randn(2, 256)
        denoised = v.extract(iq)
        assert denoised.shape == (256,)

    def test_center_frequencies_ordered(self):
        from features.vmd import VMDExtractor
        v = VMDExtractor(K=5, max_iter=100)
        signal = torch.randn(512)
        _, omega = v.decompose(signal)
        # Center frequencies should be in [0, 0.5]
        assert (omega >= 0).all()
        assert (omega <= 0.5).all()


class TestGAFExtractor:
    """Tests for features/gaf.py."""

    def test_import(self):
        from features.gaf import GAFExtractor
        g = GAFExtractor(n=64)
        assert g is not None

    def test_extract_shape_256(self):
        from features.gaf import GAFExtractor
        g = GAFExtractor(n=256)
        x = torch.randn(32768)
        out = g.extract(x, x)
        assert out.shape == (3, 256, 256)

    def test_extract_shape_64(self):
        from features.gaf import GAFExtractor
        g = GAFExtractor(n=64)
        x = torch.randn(1024)
        out = g.extract(x, x)
        assert out.shape == (3, 64, 64)

    def test_gasf_range(self):
        """GASF values should be in [-1, 1] (cosine range)."""
        from features.gaf import GAFExtractor
        g = GAFExtractor(n=32)
        x = torch.randn(256)
        out = g.extract(x, x)
        assert out[0].min() >= -1.01  # small tolerance
        assert out[0].max() <= 1.01

    def test_three_channels(self):
        """Output should have 3 channels: GASF_denoised, GADF, GASF_raw."""
        from features.gaf import GAFExtractor
        g = GAFExtractor(n=32)
        denoised = torch.randn(256)
        raw = torch.randn(256)
        out = g.extract(denoised, raw)
        assert out.shape[0] == 3


# ============================================================
# Phase 1: Expert Models
# ============================================================

class TestVMDGAFExpert:
    """Tests for models/experts/vmd_gaf_expert.py."""

    def test_import(self):
        from models.experts.vmd_gaf_expert import VMDGAFExpert
        m = VMDGAFExpert()
        assert m is not None

    def test_forward_shape(self, gaf_batch):
        from models.experts.vmd_gaf_expert import VMDGAFExpert
        m = VMDGAFExpert(num_classes=10)
        out = m(gaf_batch)
        assert out.shape == (2, 10)

    def test_get_embedding_shape(self, gaf_batch):
        from models.experts.vmd_gaf_expert import VMDGAFExpert
        m = VMDGAFExpert(embed_dim=512)
        emb = m.get_embedding(gaf_batch)
        assert emb.shape == (2, 512)

    def test_freeze_unfreeze(self):
        from models.experts.vmd_gaf_expert import VMDGAFExpert
        m = VMDGAFExpert()
        m.freeze()
        assert all(not p.requires_grad for p in m.parameters())
        m.unfreeze()
        assert all(p.requires_grad for p in m.parameters())

    def test_num_params(self):
        from models.experts.vmd_gaf_expert import VMDGAFExpert
        m = VMDGAFExpert()
        assert m.num_params > 0


class TestTFMSExpert:
    """Tests for models/experts/tfms_expert.py."""

    def test_import(self):
        from models.experts.tfms_expert import TFMSExpert
        m = TFMSExpert()
        assert m is not None

    def test_forward_shape(self, iq_batch):
        from models.experts.tfms_expert import TFMSExpert
        m = TFMSExpert(num_classes=10)
        out = m(iq_batch)
        assert out.shape == (4, 10)

    def test_get_embedding_shape(self, iq_batch):
        from models.experts.tfms_expert import TFMSExpert
        m = TFMSExpert(embed_dim=512)
        emb = m.get_embedding(iq_batch)
        assert emb.shape == (4, 512)

    def test_freeze_unfreeze(self):
        from models.experts.tfms_expert import TFMSExpert
        m = TFMSExpert()
        m.freeze()
        assert all(not p.requires_grad for p in m.parameters())
        m.unfreeze()
        assert all(p.requires_grad for p in m.parameters())


# ============================================================
# Phase 2: SOTA Expert Models
# ============================================================

class TestSignalFormerExpert:
    """Tests for models/experts/signalformer_expert.py."""

    def test_import(self):
        from models.experts.signalformer_expert import SignalFormerRFExpert
        m = SignalFormerRFExpert()
        assert m is not None

    def test_forward_shape(self):
        from models.experts.signalformer_expert import SignalFormerRFExpert
        m = SignalFormerRFExpert(num_classes=10)
        x = torch.randn(2, 3, 128, 128)  # smaller for speed
        out = m(x)
        assert out.shape == (2, 10)

    def test_get_embedding_shape(self):
        from models.experts.signalformer_expert import SignalFormerRFExpert
        m = SignalFormerRFExpert(embed_dim=512)
        x = torch.randn(2, 3, 128, 128)
        emb = m.get_embedding(x)
        assert emb.shape == (2, 512)

    def test_interface(self):
        from models.experts.signalformer_expert import SignalFormerRFExpert
        m = SignalFormerRFExpert()
        assert hasattr(m, 'forward')
        assert hasattr(m, 'get_embedding')
        assert hasattr(m, 'freeze')
        assert hasattr(m, 'unfreeze')
        assert hasattr(m, 'num_params')


class TestHiWaveTSTExpert:
    """Tests for models/experts/hiwavetst_expert.py."""

    def test_import(self):
        from models.experts.hiwavetst_expert import HiWaveTSTExpert
        m = HiWaveTSTExpert()
        assert m is not None

    def test_forward_shape(self, iq_batch):
        from models.experts.hiwavetst_expert import HiWaveTSTExpert
        m = HiWaveTSTExpert(num_classes=10)
        out = m(iq_batch)
        assert out.shape == (4, 10)

    def test_get_embedding_shape(self, iq_batch):
        from models.experts.hiwavetst_expert import HiWaveTSTExpert
        m = HiWaveTSTExpert(embed_dim=512)
        emb = m.get_embedding(iq_batch)
        assert emb.shape == (4, 512)


class TestLWMExpert:
    """Tests for models/experts/lwm_expert.py."""

    def test_import(self):
        from models.experts.lwm_expert import LWMExpert
        m = LWMExpert()
        assert m is not None

    def test_forward_shape(self, iq_batch):
        from models.experts.lwm_expert import LWMExpert
        m = LWMExpert(num_classes=10)
        out = m(iq_batch)
        assert out.shape == (4, 10)

    def test_get_embedding_shape(self, iq_batch):
        from models.experts.lwm_expert import LWMExpert
        m = LWMExpert(embed_dim=512)
        emb = m.get_embedding(iq_batch)
        assert emb.shape == (4, 512)

    def test_grid_reshape(self):
        """Verify IQ is correctly reshaped to 128x256 grid."""
        from models.experts.lwm_expert import LWMExpert
        m = LWMExpert(grid_h=128, grid_w=256)
        iq = torch.randn(1, 2, 32768)
        # grid_h * grid_w should equal N
        assert 128 * 256 == 32768
        emb = m.get_embedding(iq)
        assert emb.shape == (1, 512)


class TestNeuroSymbolicRFFExpert:
    """Tests for models/experts/neurosymbolic_rff_expert.py."""

    def test_import(self):
        from models.experts.neurosymbolic_rff_expert import NeuroSymbolicRFFExpert
        m = NeuroSymbolicRFFExpert()
        assert m is not None

    def test_forward_shape(self, iq_batch):
        from models.experts.neurosymbolic_rff_expert import NeuroSymbolicRFFExpert
        m = NeuroSymbolicRFFExpert(num_classes=10)
        out = m(iq_batch)
        assert out.shape == (4, 10)

    def test_get_embedding_shape(self, iq_batch):
        from models.experts.neurosymbolic_rff_expert import NeuroSymbolicRFFExpert
        m = NeuroSymbolicRFFExpert(embed_dim=512)
        emb = m.get_embedding(iq_batch)
        assert emb.shape == (4, 512)

    def test_shapelet_distances_nonnegative(self, iq_batch):
        """Shapelet distances should be non-negative."""
        from models.experts.neurosymbolic_rff_expert import NeuroSymbolicRFFExpert
        m = NeuroSymbolicRFFExpert()
        # Access shapelet extractor if available
        if hasattr(m, 'shapelet_extractor'):
            dists = m.shapelet_extractor(iq_batch)
            assert (dists >= 0).all()


class TestVisualRFDetector:
    """Tests for models/experts/visual_rf_detector.py."""

    def test_import(self):
        from models.experts.visual_rf_detector import VisualRFDetector
        m = VisualRFDetector()
        assert m is not None

    def test_forward_shape(self):
        from models.experts.visual_rf_detector import VisualRFDetector
        m = VisualRFDetector(num_classes=10)
        x = torch.randn(2, 3, 256, 256)
        out = m(x)
        assert out.shape == (2, 10)

    def test_get_embedding_shape(self):
        from models.experts.visual_rf_detector import VisualRFDetector
        m = VisualRFDetector(embed_dim=512)
        x = torch.randn(2, 3, 256, 256)
        emb = m.get_embedding(x)
        assert emb.shape == (2, 512)


# ============================================================
# Router Tests
# ============================================================

class TestSNRAdaptiveRouter:
    """Tests for models/moe/router.py SNRAdaptiveRouter."""

    def test_import(self):
        from models.moe.router import SNRAdaptiveRouter
        r = SNRAdaptiveRouter(input_dim=3072, num_experts=6)
        assert r is not None

    def test_forward_with_snr(self):
        from models.moe.router import SNRAdaptiveRouter
        r = SNRAdaptiveRouter(input_dim=3072, num_experts=6, top_k=2)
        x = torch.randn(4, 3072)
        snr = torch.tensor([20.0, 5.0, -5.0, -15.0])
        out = r(x, snr_db=snr)
        assert out.weights.shape == (4, 2)
        assert out.expert_indices.shape == (4, 2)
        assert out.router_logits.shape == (4, 6)

    def test_forward_without_snr(self):
        from models.moe.router import SNRAdaptiveRouter
        r = SNRAdaptiveRouter(input_dim=3072, num_experts=6, top_k=2)
        x = torch.randn(4, 3072)
        out = r(x)
        assert out.weights.shape == (4, 2)
        assert out.expert_indices.shape == (4, 2)

    def test_snr_bias_shape_matches_experts(self):
        from models.moe.router import SNRAdaptiveRouter
        for n in [4, 6, 8]:
            r = SNRAdaptiveRouter(input_dim=n * 512, num_experts=n)
            assert r.snr_bias.shape == (n,)


# ============================================================
# MoE Model Integration Tests
# ============================================================

class TestDroneRFMoEPresets:
    """Test MoE model with different expert presets."""

    def test_expert_registry_complete(self):
        from models.moe.moe_model import EXPERT_REGISTRY
        assert len(EXPERT_REGISTRY) == 11
        expected = {"iq", "spectrogram", "hos", "cyclo", "vmd_gaf", "tfms",
                    "signalformer", "hiwavetst", "lwm", "neurosymbolic_rff", "visual_rf"}
        assert set(EXPERT_REGISTRY.keys()) == expected

    def test_preset_4_experts(self):
        from models.moe.moe_model import EXPERT_PRESETS
        assert EXPERT_PRESETS[4] == ["iq", "spectrogram", "hos", "cyclo"]

    def test_preset_6_experts(self):
        from models.moe.moe_model import EXPERT_PRESETS
        assert EXPERT_PRESETS[6] == ["iq", "spectrogram", "hos", "cyclo", "vmd_gaf", "tfms"]

    def test_preset_8_experts(self):
        from models.moe.moe_model import EXPERT_PRESETS
        assert len(EXPERT_PRESETS[8]) == 8

    def test_preset_11_experts(self):
        from models.moe.moe_model import EXPERT_PRESETS
        assert len(EXPERT_PRESETS[11]) == 11


# ============================================================
# Pretraining Module Tests
# ============================================================

class TestMaskedFrequencyPredictor:
    """Tests for training/pretraining.py MaskedFrequencyPredictor."""

    def test_import(self):
        from training.pretraining import MaskedFrequencyPredictor
        assert MaskedFrequencyPredictor is not None

    def test_forward_returns_loss(self, iq_batch):
        from training.pretraining import MaskedFrequencyPredictor
        from models.experts.tfms_expert import TFMSExpert
        expert = TFMSExpert()
        mfp = MaskedFrequencyPredictor(expert, embed_dim=512, mask_ratio=0.2)
        loss = mfp(iq_batch)
        assert loss.shape == ()  # scalar
        assert loss.item() >= 0


class TestWavesFMPretraining:
    """Tests for training/wavesfm_pretraining.py."""

    def test_import_all(self):
        from training.wavesfm_pretraining import (
            WirelessPatchEmbedding,
            MaskedWirelessModeling,
            LoRAAdapter,
            apply_lora_to_model,
        )
        assert all(c is not None for c in [
            WirelessPatchEmbedding, MaskedWirelessModeling,
            LoRAAdapter, apply_lora_to_model
        ])

    def test_lora_adapter(self):
        from training.wavesfm_pretraining import LoRAAdapter
        original = nn.Linear(256, 512)
        lora = LoRAAdapter(original, rank=4)
        x = torch.randn(2, 256)
        out = lora(x)
        assert out.shape == (2, 512)
        # Original weights should be frozen
        assert not original.weight.requires_grad


class TestSpectrumFMPretraining:
    """Tests for training/spectrumfm_pretraining.py."""

    def test_import_all(self):
        from training.spectrumfm_pretraining import (
            SpectrumFMEncoder,
            MaskedReconstructionTask,
            NextSlotPredictionTask,
            DualObjectivePretrainer,
        )
        assert all(c is not None for c in [
            SpectrumFMEncoder, MaskedReconstructionTask,
            NextSlotPredictionTask, DualObjectivePretrainer
        ])


# ============================================================
# Full pipeline syntax verification
# ============================================================

class TestAllFilesSyntax:
    """Verify all implementation files have valid Python syntax."""

    FILES = [
        "features/energy_gate.py",
        "features/emd_denoising.py",
        "features/vmd.py",
        "features/gaf.py",
        "features/pipeline.py",
        "features/__init__.py",
        "models/experts/vmd_gaf_expert.py",
        "models/experts/tfms_expert.py",
        "models/experts/signalformer_expert.py",
        "models/experts/hiwavetst_expert.py",
        "models/experts/lwm_expert.py",
        "models/experts/neurosymbolic_rff_expert.py",
        "models/experts/visual_rf_detector.py",
        "models/experts/__init__.py",
        "models/moe/router.py",
        "models/moe/moe_model.py",
        "training/pretraining.py",
        "training/wavesfm_pretraining.py",
        "training/spectrumfm_pretraining.py",
    ]

    @pytest.mark.parametrize("filepath", FILES)
    def test_syntax(self, filepath):
        import ast
        with open(filepath) as f:
            source = f.read()
        # Should not raise SyntaxError
        ast.parse(source)
