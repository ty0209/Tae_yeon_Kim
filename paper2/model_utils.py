"""
Utility to replace E-MMDiT's context_embedder for EditCLIP compatibility,
and helper functions for model loading / freezing.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from core.models.transformer_emmdit import EMMDiTTransformer


def load_emmdit_for_editclip(
    emmdit_repo: str = "amd/Nitro-E",
    emmdit_ckpt: str = "Nitro-E-512px.safetensors",
    editclip_hidden_dim: int = 768,
    resolution: int = 512,
    freeze_image_stream: bool = True,
) -> EMMDiTTransformer:
    """
    Load E-MMDiT and replace context_embedder for EditCLIP.
    
    E-MMDiT original: context_embedder = Linear(2048, 768)  # Llama 3.2-1B
    EditCLIP version:  context_embedder = Linear(768, 768)   # EditCLIP ViT-B/32
    
    Args:
        emmdit_repo: HuggingFace repo for E-MMDiT weights.
        emmdit_ckpt: Checkpoint filename.
        editclip_hidden_dim: EditCLIP output dimension (768 for ViT-B/32).
        resolution: Image resolution (512 or 1024).
        freeze_image_stream: Whether to freeze image-stream parameters.
    
    Returns:
        Modified EMMDiTTransformer ready for EditCLIP fine-tuning.
    """
    transformer = EMMDiTTransformer(
        sample_size=resolution // 32,
        use_sub_attn=(resolution <= 512),
    )

    # Load pretrained weights
    state_dict = load_file(hf_hub_download(emmdit_repo, emmdit_ckpt))
    transformer.load_state_dict(state_dict)

    # Replace context_embedder
    emmdit_inner_dim = transformer.inner_dim  # 768
    old_in = transformer.context_embedder.in_features
    print(f"Replacing context_embedder: Linear({old_in}, {emmdit_inner_dim}) "
          f"-> Linear({editclip_hidden_dim}, {emmdit_inner_dim})")

    transformer.context_embedder = nn.Linear(editclip_hidden_dim, emmdit_inner_dim)
    nn.init.xavier_uniform_(transformer.context_embedder.weight)
    nn.init.zeros_(transformer.context_embedder.bias)

    if freeze_image_stream:
        apply_editclip_freeze_strategy(transformer)

    return transformer


def apply_editclip_freeze_strategy(transformer: EMMDiTTransformer):
    """
    Freeze/unfreeze strategy for EditCLIP fine-tuning:
      - Freeze: image stream, token compressor/reconstructor, AdaLN (image), pos embeds
      - Train: context_embedder, text stream (QKV, FFN, norms, AdaLN-text)
    """
    # Freeze everything first
    transformer.requires_grad_(False)

    # Unfreeze context_embedder (new)
    transformer.context_embedder.requires_grad_(True)

    # Text-stream parameter name patterns in JointTransformerBlockSingleNorm
    text_stream_patterns = [
        "add_q_proj", "add_k_proj", "add_v_proj",   # text QKV projections
        "to_add_out",                                 # text output projection
        "norm_added_q", "norm_added_k",               # text QK normalization
        "norm1_context", "norm2_context",             # text LayerNorms
        "ff_context",                                 # text FeedForward
        "scale_shift_bias_c", "scale_shift_scale_c",  # text AdaLN-affine params
    ]

    trainable_count = 0
    frozen_count = 0

    for name, param in transformer.named_parameters():
        if any(pattern in name for pattern in text_stream_patterns):
            param.requires_grad_(True)
            trainable_count += param.numel()
        elif "context_embedder" in name:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    total = trainable_count + frozen_count
    print(f"Freeze strategy applied:")
    print(f"  Trainable: {trainable_count:,} ({100*trainable_count/total:.1f}%)")
    print(f"  Frozen:    {frozen_count:,} ({100*frozen_count/total:.1f}%)")

    return transformer


def count_parameters(model: nn.Module):
    """Print parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")
    return trainable, total
