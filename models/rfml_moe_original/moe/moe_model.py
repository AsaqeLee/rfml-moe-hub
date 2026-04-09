"""Complete Mixture-of-Experts model for drone RF detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.experts.iq_expert import IQExpert
from models.experts.spectrogram_expert import SpectrogramExpert
from models.experts.hos_expert import HOSExpert
from models.experts.cyclo_expert import CycloExpert
from models.experts.vmd_gaf_expert import VMDGAFExpert
from models.experts.tfms_expert import TFMSExpert
from models.experts.signalformer_expert import SignalFormerRFExpert
from models.experts.hiwavetst_expert import HiWaveTSTExpert
from models.experts.lwm_expert import LWMExpert
from models.experts.neurosymbolic_rff_expert import NeuroSymbolicRFFExpert
from models.experts.visual_rf_detector import VisualRFDetector
from models.fusion.cross_attention import CrossAttentionFusion
from models.moe.router import ExpertChoiceRouter, SNRAdaptiveRouter
from models.moe.load_balance import compute_expert_utilization, router_z_loss

# Registry of all available experts and their input types
EXPERT_REGISTRY = {
    # Phase 0: Original experts
    "iq": {"class": IQExpert, "input_key": "iq"},
    "spectrogram": {"class": SpectrogramExpert, "input_key": "spectrogram"},
    "hos": {"class": HOSExpert, "input_key": "hos"},
    "cyclo": {"class": CycloExpert, "input_key": "cyclo"},
    # Phase 1: Paper integration experts
    "vmd_gaf": {"class": VMDGAFExpert, "input_key": "vmd_gaf"},
    "tfms": {"class": TFMSExpert, "input_key": "iq"},
    # Phase 2: SOTA model experts
    "signalformer": {"class": SignalFormerRFExpert, "input_key": "spectrogram"},
    "hiwavetst": {"class": HiWaveTSTExpert, "input_key": "iq"},
    "lwm": {"class": LWMExpert, "input_key": "iq"},
    "neurosymbolic_rff": {"class": NeuroSymbolicRFFExpert, "input_key": "iq"},
    "visual_rf": {"class": VisualRFDetector, "input_key": "spectrogram"},
}

# Preset expert configurations
EXPERT_PRESETS = {
    4: ["iq", "spectrogram", "hos", "cyclo"],
    6: ["iq", "spectrogram", "hos", "cyclo", "vmd_gaf", "tfms"],
    8: ["iq", "spectrogram", "hos", "cyclo", "vmd_gaf", "tfms", "signalformer", "hiwavetst"],
    11: list(EXPERT_REGISTRY.keys()),
}


class SharedExpert(nn.Module):
    """DeepSeek-style shared expert that is always active.

    Processes the concatenation of all expert embeddings and produces
    a shared representation added to the fused output.

    Args:
        input_dim: Total concatenated input dimension (num_experts * embed_dim).
        embed_dim: Output embedding dimension.
    """

    def __init__(self, input_dim: int = 2048, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DroneRFMoE(nn.Module):
    """Mixture-of-Experts model for hierarchical drone RF classification.

    Combines six domain-specific experts (IQ, Spectrogram, HOS, Cyclostationary,
    VMD-GAF, TFMS) with SNR-adaptive Expert Choice routing, cross-attention fusion,
    an optional shared expert, and hierarchical classification heads.

    The model supports both 4-expert (legacy) and 6-expert (extended) configurations
    controlled by the ``num_experts`` parameter or config file.

    Supports two construction patterns:
        model = DroneRFMoE(config)           # from Config object
        model = DroneRFMoE(embed_dim=512, num_experts=6, ...)  # explicit params
    """

    def __init__(
        self,
        config=None,
        embed_dim: int = 512,
        num_experts: int = 6,
        top_k: int = 2,
        num_classes_l1: int = 2,
        num_classes_l2: int = 15,
        num_classes_l3: int = 50,
        use_shared_expert: bool = True,
        use_snr_routing: bool = True,
        dropout: float = 0.1,
        z_loss_coeff: float = 0.002,
    ):
        super().__init__()

        # If config object provided, extract parameters from it
        if config is not None:
            moe_cfg = config.get_nested("moe", {})
            cls_cfg = config.get_nested("classification.hierarchy", {})
            embed_dim = int(moe_cfg.get("router_dim", 512))
            num_experts = int(moe_cfg.get("num_experts", 6))
            top_k = int(moe_cfg.get("top_k", 2))
            num_classes_l1 = int(cls_cfg.get("level1_classes", 2))
            num_classes_l2 = int(cls_cfg.get("level2_classes", 15))
            num_classes_l3 = int(cls_cfg.get("level3_classes", 50))
            use_shared_expert = bool(moe_cfg.get("shared_expert", True))
            use_snr_routing = bool(moe_cfg.get("snr_adaptive_routing", True))
            z_loss_coeff = float(
                moe_cfg.get("load_balancing", {}).get("z_loss_coeff", 0.002)
            )

        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        self.use_snr_routing = use_snr_routing
        self.z_loss_coeff = z_loss_coeff

        # Domain-specific experts via registry-driven configuration
        if num_experts in EXPERT_PRESETS:
            self.expert_keys = EXPERT_PRESETS[num_experts]
        else:
            # Fallback: use first N experts from full registry
            self.expert_keys = list(EXPERT_REGISTRY.keys())[:num_experts]

        self.experts = nn.ModuleDict({
            key: EXPERT_REGISTRY[key]["class"]()
            for key in self.expert_keys
        })
        self._expert_input_map = {
            key: EXPERT_REGISTRY[key]["input_key"]
            for key in self.expert_keys
        }

        # Router: SNR-adaptive (6 experts) or standard (4 experts)
        input_dim = num_experts * embed_dim  # 6 * 512 = 3072
        if use_snr_routing and num_experts == 6:
            self.router = SNRAdaptiveRouter(
                input_dim=input_dim,
                num_experts=num_experts,
                top_k=top_k,
            )
        else:
            self.router = ExpertChoiceRouter(
                input_dim=input_dim,
                num_experts=num_experts,
                top_k=top_k,
            )

        # Cross-attention fusion (1 layer for 6 experts to control O(M²) cost)
        fusion_layers = 1 if num_experts >= 6 else 2
        self.fusion = CrossAttentionFusion(
            num_modalities=num_experts,
            embed_dim=embed_dim,
            num_layers=fusion_layers,
        )

        # Optional shared expert (DeepSeek-style, always active)
        self.shared_expert = None
        if use_shared_expert:
            self.shared_expert = SharedExpert(input_dim=input_dim, embed_dim=embed_dim)

        # Hierarchical classification heads
        self.head_l1 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes_l1),
        )
        self.head_l2 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes_l2),
        )
        self.head_l3 = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes_l3),
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor] | None = None,
        *,
        iq: torch.Tensor | None = None,
        spectrogram: torch.Tensor | None = None,
        hos: torch.Tensor | None = None,
        cyclo: torch.Tensor | None = None,
        vmd_gaf: torch.Tensor | None = None,
        tfms_iq: torch.Tensor | None = None,
        snr_db: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the full MoE pipeline.

        Accepts either a dict or keyword arguments:
            model({"iq": t1, "spectrogram": t2, ..., "vmd_gaf": t5, "snr_db": s})
            model(iq=t1, spectrogram=t2, ..., vmd_gaf=t5, snr_db=s)

        Returns:
            Dict with keys matching both naming conventions for compatibility:
                "level1" / "logits_binary": logits (batch, num_classes_l1)
                "level2" / "logits_type":   logits (batch, num_classes_l2)
                "level3" / "logits_full":   logits (batch, num_classes_l3)
                "router_logits": raw router logits (batch, num_experts)
                "expert_indices": selected expert indices (batch, top_k)
                "router_probs": routing probabilities (batch, num_experts)
                "aux_loss": auxiliary load-balancing loss (scalar)
        """
        # Build inputs dict from either pattern
        if inputs is None:
            inputs = {
                "iq": iq, "spectrogram": spectrogram,
                "hos": hos, "cyclo": cyclo,
                "vmd_gaf": vmd_gaf, "tfms": tfms_iq,
                "snr_db": snr_db,
            }

        # Extract SNR from inputs dict (not an expert input, use .get to avoid mutation)
        snr_db_tensor = inputs.get("snr_db", None) if isinstance(inputs, dict) else snr_db

        # Extract embeddings from each expert via get_embedding()
        # Uses _expert_input_map to route correct input tensor to each expert
        expert_embeddings = []
        for key in self.expert_keys:
            input_key = self._expert_input_map[key]
            expert_input = inputs.get(input_key)
            if expert_input is None:
                raise ValueError(
                    f"Expert '{key}' requires input '{input_key}' but it was not provided. "
                    f"Available keys: {list(inputs.keys())}"
                )
            emb = self.experts[key].get_embedding(expert_input)  # (batch, embed_dim)
            expert_embeddings.append(emb)

        # Concatenate for routing
        concat_emb = torch.cat(expert_embeddings, dim=-1)  # (batch, num_experts*embed_dim)

        # Route (with optional SNR conditioning for SNRAdaptiveRouter)
        if isinstance(self.router, SNRAdaptiveRouter):
            router_out = self.router(concat_emb, snr_db=snr_db_tensor)
        else:
            router_out = self.router(concat_emb)

        # Apply routing weights to expert embeddings
        weighted_embeddings = []
        for i in range(self.num_experts):
            mask = (router_out.expert_indices == i).float()  # (batch, top_k)
            weight = (mask * router_out.weights).sum(dim=-1, keepdim=True)  # (batch, 1)
            weighted_embeddings.append(expert_embeddings[i] * weight)

        # Cross-attention fusion on weighted embeddings
        fused = self.fusion(weighted_embeddings)  # (batch, embed_dim)

        # Add shared expert output if enabled
        if self.shared_expert is not None:
            shared_out = self.shared_expert(concat_emb)  # (batch, embed_dim)
            fused = fused + shared_out

        # Hierarchical classification
        level1 = self.head_l1(fused)
        level2 = self.head_l2(fused)
        level3 = self.head_l3(fused)

        # Compute auxiliary load-balancing loss (router Z-loss)
        aux_loss = router_z_loss(router_out.router_logits) * self.z_loss_coeff

        return {
            # Primary keys
            "level1": level1,
            "level2": level2,
            "level3": level3,
            # Aliases for trainer compatibility
            "logits_binary": level1,
            "logits_type": level2,
            "logits_full": level3,
            # Routing info
            "router_logits": router_out.router_logits,
            "expert_indices": router_out.expert_indices,
            "router_probs": router_out.load_balance_stats["router_probs"],
            "aux_loss": aux_loss,
        }

    def freeze_experts(self) -> None:
        """Freeze all expert parameters (for fine-tuning heads/router only)."""
        for expert in self.experts.values():
            for param in expert.parameters():
                param.requires_grad = False

    def unfreeze_experts(self) -> None:
        """Unfreeze all expert parameters."""
        for expert in self.experts.values():
            for param in expert.parameters():
                param.requires_grad = True

    def get_expert_utilization(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """Compute per-expert utilization fractions from routing indices.

        Args:
            expert_indices: Expert indices tensor (batch, top_k).

        Returns:
            Tensor of shape (num_experts,) with usage fractions.
        """
        return compute_expert_utilization(expert_indices, self.num_experts)
