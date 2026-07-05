"""
Metrics for EditCLIP-style benchmark comparison.

  LPIPS  : perceptual distance between generated edit and ground-truth edit.
           Lower is better. Implemented via the `lpips` package (AlexNet).
  SSIM   : structural similarity between generated edit and ground-truth edit.
           Higher is better. Implemented via skimage.
  EC2EC  : EditCLIP-to-EditCLIP cosine similarity. Encode the GT pair
           (input_gt, target_gt) and the candidate pair (test_input, output)
           through the EditCLIP 6-channel-concat vision tower, mean-pool over
           patch tokens, and compare. Higher is better. This is EditCLIP's
           signature edit-fidelity metric — it measures how closely the
           model-applied transformation MATCHES the exemplar's
           transformation, not just pixel/structure agreement.
  CLIP   : Standard CLIP text-image cosine similarity between a target text
           prompt and the model output. Uses OpenAI CLIP ViT-L/14 by default
           (independent of EditCLIP). Higher is better. Lets you compare a
           text-conditioned method (Llama+Nitro-E) against an exemplar-
           conditioned method (EditCLIP+Nitro-E) under a SHARED text-prompt
           reference — even though the exemplar method never sees the text.

The first four metrics operate on PIL.Image inputs at a common evaluation
resolution (default 256x256, matching the IP2P training resolution and the
EditCLIP paper's eval protocol). For higher-resolution outputs the caller
must resize first.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def to_eval_size(img: Image.Image, size: int = 256) -> Image.Image:
    return img.convert("RGB").resize((size, size), Image.LANCZOS)


def to_tensor_01(img: Image.Image) -> torch.Tensor:
    """[H, W, 3] PIL -> [1, 3, H, W] in [0, 1] float."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _lpips_model(net: str = "alex"):
    import lpips
    model = lpips.LPIPS(net=net).to(_DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def lpips_pair(a: Image.Image, b: Image.Image, size: int = 256,
                net: str = "alex") -> float:
    a = to_eval_size(a, size)
    b = to_eval_size(b, size)
    ta = to_tensor_01(a) * 2.0 - 1.0   # lpips expects [-1, 1]
    tb = to_tensor_01(b) * 2.0 - 1.0
    ta = ta.to(_DEVICE)
    tb = tb.to(_DEVICE)
    model = _lpips_model(net)
    d = model(ta, tb).item()
    return float(d)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def ssim_pair(a: Image.Image, b: Image.Image, size: int = 256) -> float:
    """Returns SSIM in [-1, 1], typically [0, 1] for natural images."""
    from skimage.metrics import structural_similarity as _ssim
    a = to_eval_size(a, size)
    b = to_eval_size(b, size)
    arr_a = np.asarray(a, dtype=np.float32) / 255.0
    arr_b = np.asarray(b, dtype=np.float32) / 255.0
    return float(_ssim(arr_a, arr_b, channel_axis=2, data_range=1.0))


# ---------------------------------------------------------------------------
# EC2EC (EditCLIP-to-EditCLIP)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _editclip(editclip_path: str):
    """Load the FULL CLIPModel — we need both vision_model (for the 6-ch
    encoder forward) AND visual_projection (the CLIP projection head that
    maps to the shared embedding space used for cosine similarity).
    """
    from transformers import CLIPModel, AutoProcessor
    model = CLIPModel.from_pretrained(editclip_path)
    model = model.to(_DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(editclip_path)
    return model, proc


@torch.no_grad()
def editclip_pair_feature(src: Image.Image, dst: Image.Image,
                           editclip_path: str,
                           use_projection: bool = True) -> torch.Tensor:
    """6-channel concat -> EditCLIP visual encoder.

    With `use_projection=True` (default, matches EditCLIP paper convention),
    the encoder's pooled output passes through `visual_projection`, yielding
    the CLIP-shared 768-d (for ViT-L/14) embedding used for cosine similarity.
    Without projection, returns mean-pooled patch tokens in the 1024-d raw
    vision space (legacy, biased high).

    Returns a [D]-dim feature vector on CPU (float).
    """
    model, proc = _editclip(editclip_path)
    src = src.convert("RGB")
    dst = dst.convert("RGB")
    o = proc(images=src, return_tensors="pt")
    e = proc(images=dst, return_tensors="pt")
    concat = torch.cat([o.pixel_values, e.pixel_values], dim=1).to(_DEVICE)
    vis = model.vision_model(concat)
    if use_projection:
        # pooler_output is the CLS-derived feature; visual_projection maps to
        # the CLIP-shared embedding space.
        proj = model.visual_projection(vis.pooler_output)  # [1, D_proj]
        return proj.squeeze(0).float().cpu()
    feats = vis.last_hidden_state[:, 1:, :]
    pooled = feats.mean(dim=1).squeeze(0)
    return pooled.float().cpu()


@torch.no_grad()
def ec2ec_pair(gt_src: Image.Image, gt_dst: Image.Image,
                cand_src: Image.Image, cand_dst: Image.Image,
                editclip_path: str) -> float:
    """Cosine similarity between two EditCLIP pair-features.

    gt   = (input_gt, target_gt)   = the reference pair describing the edit
    cand = (cand_src, cand_dst)    = candidate pair (typically test_input, output)

    NOTE: if gt_src == cand_src (same image), the 6-channel concat's first
    3 channels are identical between the two pair-features, which makes the
    cosine artificially high (typically >0.85). That's the "ec2ec_to_gt"
    interpretation. To get the EditCLIP-paper edit-direction metric, pass
    an EXEMPLAR pair as gt (different source from candidate). See
    `ec2ec_edit_direction` below.
    """
    f_gt = editclip_pair_feature(gt_src, gt_dst, editclip_path)
    f_cd = editclip_pair_feature(cand_src, cand_dst, editclip_path)
    cs = F.cosine_similarity(f_gt.unsqueeze(0), f_cd.unsqueeze(0), dim=-1)
    return float(cs.item())


@torch.no_grad()
def ec2ec_edit_direction(exemplar_src: Image.Image,
                          exemplar_ed: Image.Image,
                          test_input: Image.Image,
                          model_output: Image.Image,
                          editclip_path: str) -> float:
    """EditCLIP paper's EC2EC metric.

    Compares the edit-direction encoded in a TRAIN exemplar pair against
    the edit-direction encoded in (test_input, model_output). Since the
    two pairs have DIFFERENT source images, the cosine is no longer
    dominated by source identity and meaningfully ranges over [~0.3, ~0.6]
    for natural pairs. EditCLIP+IP2P in the paper reports ~0.477.
    """
    return ec2ec_pair(exemplar_src, exemplar_ed,
                        test_input, model_output, editclip_path)


# ---------------------------------------------------------------------------
# CLIP Score (standard text-image cosine, NOT EditCLIP)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _clip_text_image(model_name: str = "openai/clip-vit-large-patch14"):
    """Load a standard CLIP (text+vision) tower. Distinct from EditCLIP, which
    is a 6-channel-image-only variant.
    """
    from transformers import CLIPModel, AutoProcessor
    model = CLIPModel.from_pretrained(model_name).to(_DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(model_name)
    return model, proc


@torch.no_grad()
def clip_score(prompt: str, output_image: Image.Image,
                 clip_model_name: str = "openai/clip-vit-large-patch14",
                 size: int = 256,
                 scale_100: bool = True) -> float:
    """Cosine similarity between CLIP text features (prompt) and image
    features (model output). Returned in the conventional 0-100 range when
    `scale_100=True` (paper convention). Higher = better text alignment.
    """
    model, proc = _clip_text_image(clip_model_name)
    img = output_image.convert("RGB").resize((size, size), Image.LANCZOS)
    inputs = proc(text=[prompt], images=[img], return_tensors="pt",
                   padding=True, truncation=True)
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
    # Single forward through the joint model — gives us aligned text/image
    # embeddings without the get_*_features wrappers (which return different
    # types across transformers versions).
    out = model(input_ids=inputs["input_ids"],
                  attention_mask=inputs["attention_mask"],
                  pixel_values=inputs["pixel_values"])
    t = out.text_embeds
    v = out.image_embeds
    t = F.normalize(t, dim=-1)
    v = F.normalize(v, dim=-1)
    cs = (t * v).sum(dim=-1).item()
    return float(cs * 100.0 if scale_100 else cs)


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all(input_test: Image.Image,
                  gt_edit: Image.Image,
                  model_output: Image.Image,
                  editclip_path: str,
                  size: int = 256,
                  lpips_net: str = "alex",
                  clip_prompt: Optional[str] = None,
                  clip_model_name: str = "openai/clip-vit-large-patch14",
                  exemplar_src: Optional[Image.Image] = None,
                  exemplar_ed: Optional[Image.Image] = None,
                  ) -> dict:
    """Return all available metrics for one (gt, output) comparison.

    EC2EC variants
    --------------
    Two flavours are reported when possible:

      * `ec2ec_to_gt` : cosine((test_in, gt_edit), (test_in, output)).
                         Same source on both sides => biased HIGH (~0.85+)
                         by source-identity overlap. Useful for tracking
                         output->GT closeness internally but NOT what the
                         EditCLIP paper reports.

      * `ec2ec_edit_dir` : cosine((exemplar_src, exemplar_ed),
                                    (test_in, output)).
                             EditCLIP paper convention. Sources differ so
                             cosine reflects edit-direction similarity, not
                             source identity. EditCLIP+IP2P in the paper
                             reports ~0.477. Only computed when an
                             exemplar pair is provided.

    `clip_prompt` is optional. If provided, CLIP Score is computed against it.
    """
    out = {
        "lpips": lpips_pair(gt_edit, model_output, size=size, net=lpips_net),
        "ssim":  ssim_pair(gt_edit, model_output, size=size),
        # Legacy (biased-high) metric kept for continuity.
        "ec2ec_to_gt": ec2ec_pair(input_test, gt_edit,
                                    input_test, model_output,
                                    editclip_path=editclip_path),
    }
    # EditCLIP paper's edit-direction EC2EC — only when an exemplar is given.
    if exemplar_src is not None and exemplar_ed is not None:
        out["ec2ec_edit_dir"] = ec2ec_edit_direction(
            exemplar_src, exemplar_ed,
            input_test, model_output,
            editclip_path=editclip_path,
        )
    if clip_prompt is not None:
        out["clip_score"] = clip_score(clip_prompt, model_output,
                                          clip_model_name=clip_model_name,
                                          size=size)
    return out
