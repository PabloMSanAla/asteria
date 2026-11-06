"""
Target Generation, Losses, and Metrics for UNet++ Center Offset Voting.
=======================================================================

Provides:
1. Target generation:
   - compute_instance_centroids
   - generate_center_offset_targets
   - generate_center_offset_batch
2. Multi-task loss functions:
   - CenterFocalLoss: Penalty-reduced Gaussian focal loss for keypoint heatmaps.
   - MaskedSmoothL1Loss: Offset regression restricted to foreground pixels.
   - AngularCosineDistanceLoss: Directional cosine loss along diffraction spikes.
   - CenterOffsetCompositeLoss: Multi-task weighted objective.
3. Evaluation metrics:
   - evaluate_center_detection: Precision, Recall, F1, and MAE for centroid detection.
   - compute_offset_epe: Endpoint error (EPE) for 2D displacement vectors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from asteria.inference.center_offset import extract_center_peaks


def compute_instance_centroids(
    instance_mask: np.ndarray,
    min_area: int = 1,
) -> Dict[int, Tuple[float, float, int]]:
    """
    Extract geometric center of mass (cy, cx) and pixel area for each instance.

    Parameters
    ----------
    instance_mask : np.ndarray
        2D integer array (H, W) where 0 is background and k > 0 are instance IDs.
    min_area : int, default 1
        Minimum pixel count to retain an instance.

    Returns
    -------
    Dict[int, Tuple[float, float, int]]
        Mapping from instance_id to (cy, cx, area).
    """
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]
    centroids: Dict[int, Tuple[float, float, int]] = {}

    for inst_id in unique_ids:
        ys, xs = np.where(instance_mask == inst_id)
        area = len(ys)
        if area < min_area:
            continue
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        centroids[int(inst_id)] = (cy, cx, area)

    return centroids


def generate_center_offset_targets(
    instance_mask: np.ndarray,
    sigma: float = 2.0,
    min_area: int = 1,
    adaptive_sigma: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    Generate foreground mask, Gaussian center heatmap, and (dy, dx) offset field (NumPy).

    Parameters
    ----------
    instance_mask : np.ndarray
        Integer array of shape (H, W) with instance IDs (0 is background).
    sigma : float, default 2.0
        Standard deviation for Gaussian center peaks in pixels.
    min_area : int, default 1
        Minimum instance pixel area.
    adaptive_sigma : bool, default False
        If True, scales sigma with object effective radius.

    Returns
    -------
    Tuple containing:
        - fg_mask : (H, W) binary float32 mask
        - center_map : (H, W) float32 Gaussian heatmap in [0.0, 1.0]
        - offset_map : (2, H, W) float32 displacement vector field [dy, dx]
        - info_list : metadata list with centroid coordinates and radii
    """
    h, w = instance_mask.shape
    fg_mask = (instance_mask > 0).astype(np.float32)
    center_map = np.zeros((h, w), dtype=np.float32)
    offset_map = np.zeros((2, h, w), dtype=np.float32)

    centroids = compute_instance_centroids(instance_mask, min_area=min_area)
    info_list: List[Dict[str, float]] = []

    for inst_id, (cy, cx, area) in centroids.items():
        ys, xs = np.where(instance_mask == inst_id)

        # Target offsets: vector pointing from (y, x) toward (cy, cx)
        # dy = cy - y, dx = cx - x
        offset_map[0, ys, xs] = (cy - ys).astype(np.float32)
        offset_map[1, ys, xs] = (cx - xs).astype(np.float32)

        inst_sigma = sigma
        if adaptive_sigma:
            r_eff = np.sqrt(area / np.pi)
            inst_sigma = max(1.0, min(float(r_eff * 0.4), 8.0))

        # Splat Gaussian center heatmap
        radius = int(np.ceil(3 * inst_sigma))
        y_min = max(0, int(np.floor(cy - radius)))
        y_max = min(h, int(np.ceil(cy + radius + 1)))
        x_min = max(0, int(np.floor(cx - radius)))
        x_max = min(w, int(np.ceil(cx + radius + 1)))

        sub_y, sub_x = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (sub_y - cy) ** 2 + (sub_x - cx) ** 2
        gaussian = np.exp(-dist_sq / (2.0 * inst_sigma ** 2)).astype(np.float32)

        center_map[y_min:y_max, x_min:x_max] = np.maximum(
            center_map[y_min:y_max, x_min:x_max], gaussian
        )

        # Ensure exact center pixel is 1.0
        cy_int = int(np.clip(np.round(cy), 0, h - 1))
        cx_int = int(np.clip(np.round(cx), 0, w - 1))
        center_map[cy_int, cx_int] = 1.0

        info_list.append({"inst_id": float(inst_id), "cy": cy, "cx": cx, "area": float(area)})

    return fg_mask, center_map, offset_map, info_list


def generate_center_offset_batch(
    instance_masks: Union[np.ndarray, torch.Tensor],
    sigma: float = 2.0,
    adaptive_sigma: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate multi-task targets for a batch of instance masks.

    Parameters
    ----------
    instance_masks : (B, H, W) integer masks
    sigma : float, default 2.0
    adaptive_sigma : bool, default False

    Returns
    -------
    Tuple containing:
        - fg_batch : (B, 1, H, W)
        - center_batch : (B, 1, H, W)
        - offset_batch : (B, 2, H, W)
    """
    if isinstance(instance_masks, torch.Tensor):
        m_np = instance_masks.detach().cpu().numpy()
    else:
        m_np = np.asarray(instance_masks)

    b, h, w = m_np.shape
    fg_list, ctr_list, off_list = [], [], []

    for i in range(b):
        fg, ctr, off, _ = generate_center_offset_targets(
            m_np[i], sigma=sigma, adaptive_sigma=adaptive_sigma
        )
        fg_list.append(fg)
        ctr_list.append(ctr)
        off_list.append(off)

    fg_tensor = torch.from_numpy(np.stack(fg_list, axis=0)).unsqueeze(1).float()
    ctr_tensor = torch.from_numpy(np.stack(ctr_list, axis=0)).unsqueeze(1).float()
    off_tensor = torch.from_numpy(np.stack(off_list, axis=0)).float()

    return fg_tensor, ctr_tensor, off_tensor


class DiceLoss(nn.Module):
    """Dice loss for foreground binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(pred_logits)
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1).float()

        intersection = (probs_flat * targets_flat).sum()
        total = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice


class CenterFocalLoss(nn.Module):
    """
    Penalty-reduced Gaussian focal loss for center keypoints (CenterNet).

    Paper: Zhou et al., "Objects as Points" (2019).
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        eps: float = 1e-6,
        logit_clamp: float = 15.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.logit_clamp = float(logit_clamp)

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = pred_logits.float()
        if self.logit_clamp > 0:
            logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)

        pos_mask = targets.eq(1.0).float()
        neg_mask = targets.lt(1.0).float()

        neg_weights = torch.pow(1.0 - targets, self.beta)
        probs = torch.sigmoid(logits)

        # Clamped log probabilities
        pos_loss = torch.log(torch.clamp(probs, min=self.eps, max=1.0)) * torch.pow(1.0 - probs, self.alpha) * pos_mask
        neg_loss = torch.log(torch.clamp(1.0 - probs, min=self.eps, max=1.0)) * torch.pow(probs, self.alpha) * neg_weights * neg_mask

        num_pos = pos_mask.sum()
        pos_loss_sum = pos_loss.sum()
        neg_loss_sum = neg_loss.sum()

        if num_pos > 0:
            loss = -(pos_loss_sum + neg_loss_sum) / num_pos
        else:
            loss = -neg_loss_sum / max(1.0, targets.shape[0])

        return loss


class MaskedSmoothL1Loss(nn.Module):
    """Smooth L1 Loss evaluated strictly over foreground pixels."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pred_offset: torch.Tensor,
        target_offset: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = fg_mask.expand_as(pred_offset) > 0.5
        if not mask.any():
            return torch.tensor(0.0, device=pred_offset.device, requires_grad=True)

        diff = pred_offset[mask] - target_offset[mask]
        abs_diff = torch.abs(diff)

        smooth_l1 = torch.where(
            abs_diff < self.beta,
            0.5 * (abs_diff ** 2) / self.beta,
            abs_diff - 0.5 * self.beta,
        )
        return smooth_l1.mean()


class AngularCosineDistanceLoss(nn.Module):
    """
    Directional cosine distance loss enforcing offset alignment along diffraction spikes.
    """

    def __init__(self, eps: float = 1e-6, min_radius: float = 3.0):
        super().__init__()
        self.eps = eps
        self.min_radius = min_radius

    def forward(
        self,
        pred_offset: torch.Tensor,
        target_offset: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = pred_offset.shape
        target_norm = torch.sqrt(torch.sum(target_offset ** 2, dim=1, keepdim=True) + self.eps)

        valid_mask = (fg_mask > 0.5) & (target_norm >= self.min_radius)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_offset.device, requires_grad=True)

        pred_norm = torch.sqrt(torch.sum(pred_offset ** 2, dim=1, keepdim=True) + self.eps)
        dot_prod = torch.sum(pred_offset * target_offset, dim=1, keepdim=True)
        cos_sim = dot_prod / (pred_norm * target_norm)

        cos_dist = 1.0 - torch.clamp(cos_sim, -1.0, 1.0)
        return cos_dist[valid_mask].mean()


class CenterOffsetCompositeLoss(nn.Module):
    """
    Composite Multi-Task loss for Center Offset Voting instance segmentation.
    Combines:
    - Semantic Loss (BCE + Dice)
    - Center Keypoint Focal Loss (CenterFocalLoss)
    - Offset Vector Regression (MaskedSmoothL1Loss)
    - Diffraction Spike Ray-Alignment (AngularCosineDistanceLoss)
    """

    def __init__(
        self,
        semantic_weight: float = 1.0,
        center_weight: float = 2.0,
        offset_weight: float = 1.0,
        angle_weight: float = 0.5,
        smooth_l1_beta: float = 1.0,
    ):
        super().__init__()
        self.semantic_weight = semantic_weight
        self.center_weight = center_weight
        self.offset_weight = offset_weight
        self.angle_weight = angle_weight

        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.center_loss = CenterFocalLoss()
        self.offset_loss = MaskedSmoothL1Loss(beta=smooth_l1_beta)
        self.angle_loss = AngularCosineDistanceLoss()

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        target_fg: torch.Tensor,
        target_center: torch.Tensor,
        target_offset: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred_sem = preds["semantic"]
        pred_ctr = preds["center"]
        pred_off = preds["offset"]

        # Semantic loss: BCE + Dice
        l_bce = self.bce_loss(pred_sem, target_fg)
        l_dice = self.dice_loss(pred_sem, target_fg)
        l_sem = l_bce + l_dice

        # Center heatmap loss
        l_ctr = self.center_loss(pred_ctr, target_center)

        # Offset regression loss
        l_off = self.offset_loss(pred_off, target_offset, target_fg)

        # Angular alignment loss
        l_ang = self.angle_loss(pred_off, target_offset, target_fg)

        total_loss = (
            self.semantic_weight * l_sem
            + self.center_weight * l_ctr
            + self.offset_weight * l_off
            + self.angle_weight * l_ang
        )

        return {
            "loss": total_loss,
            "loss_semantic": l_sem,
            "loss_center": l_ctr,
            "loss_offset": l_off,
            "loss_angle": l_ang,
        }


def evaluate_center_detection(
    pred_peaks: List[Tuple[float, float, float]],
    gt_centroids: List[Tuple[float, float]],
    tolerance: float = 2.0,
) -> Dict[str, float]:
    """
    Evaluate center detection performance via bipartite greedy distance matching.
    """
    if len(pred_peaks) == 0 and len(gt_centroids) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mae": 0.0}
    if len(pred_peaks) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": 0.0}
    if len(gt_centroids) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "mae": 0.0}

    preds = np.array([[p[0], p[1]] for p in pred_peaks], dtype=np.float32)
    gts = np.array(gt_centroids, dtype=np.float32)

    dists = np.sqrt(((preds[:, None, :] - gts[None, :, :]) ** 2).sum(axis=-1))

    matched_gt = set()
    matched_pred = set()
    errors = []

    pairs = []
    for pi in range(len(preds)):
        for gi in range(len(gts)):
            pairs.append((dists[pi, gi], pi, gi))
    pairs.sort(key=lambda x: x[0])

    for d, pi, gi in pairs:
        if pi not in matched_pred and gi not in matched_gt:
            if d <= tolerance:
                matched_pred.add(pi)
                matched_gt.add(gi)
                errors.append(d)

    tp = len(matched_gt)
    fp = len(preds) - tp
    fn = len(gts) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mae = float(np.mean(errors)) if len(errors) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mae": float(mae),
    }


def compute_offset_epe(
    pred_offsets: Union[np.ndarray, torch.Tensor],
    target_offsets: Union[np.ndarray, torch.Tensor],
    fg_mask: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute End-Point Error (EPE) on 2D displacement vectors over foreground pixels."""
    if isinstance(pred_offsets, torch.Tensor):
        p_np = pred_offsets.detach().float().cpu().numpy()
        t_np = target_offsets.detach().float().cpu().numpy()
        m_np = fg_mask.detach().float().cpu().numpy()
    else:
        p_np = np.asarray(pred_offsets, dtype=np.float32)
        t_np = np.asarray(target_offsets, dtype=np.float32)
        m_np = np.asarray(fg_mask, dtype=np.float32)

    # Broadcast or squeeze masks
    if p_np.ndim == 4:
        # (B, 2, H, W) and (B, 1, H, W)
        mask = m_np[:, 0] > 0.5 if m_np.ndim == 4 else m_np > 0.5
        dy_err = p_np[:, 0][mask] - t_np[:, 0][mask]
        dx_err = p_np[:, 1][mask] - t_np[:, 1][mask]
    else:
        # (2, H, W) and (H, W)
        mask = m_np.squeeze() > 0.5
        dy_err = p_np[0][mask] - t_np[0][mask]
        dx_err = p_np[1][mask] - t_np[1][mask]

    if len(dy_err) == 0:
        return 0.0

    epe = np.sqrt(dy_err ** 2 + dx_err ** 2)
    return float(np.mean(epe))
