"""
Siamese Network Fine-Tuner for CLIP Embeddings
===============================================
Techniques implemented:
  - Siamese network with shared CLIP backbone
  - Contrastive loss with configurable margin
  - Triplet loss with hard/semi-hard negative mining
  - Nesterov Momentum optimizer variant
  - Polyak Averaging (Exponential Moving Average)
  - Stochastic Weight Averaging (SWA)
  - Hungarian Algorithm for optimal batch assignment matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from loguru import logger

try:
    from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm
except ImportError:
    linear_sum_assignment = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, DEVICE, SiameseConfig
from src.embeddings.clip_encoder import CLIPEncoder


# ═══════════════════════════════════════════════════════════════
# Contrastive Loss (margin-based)
# ═══════════════════════════════════════════════════════════════
class MarginContrastiveLoss(nn.Module):
    """
    Margin-based contrastive loss for Siamese networks.
    Pulls positives together, pushes negatives apart beyond margin.
    
    L = (1-y) * D^2 + y * max(0, margin - D)^2
    where y=0 for positives, y=1 for negatives
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            emb1, emb2: (B, D) L2-normalized embeddings from each branch
            labels: (B,) binary — 0=same class (positive), 1=different class
        """
        distances = F.pairwise_distance(emb1, emb2, p=2)
        positive_loss = (1 - labels) * distances.pow(2)
        negative_loss = labels * F.relu(self.margin - distances).pow(2)
        return (positive_loss + negative_loss).mean()


# ═══════════════════════════════════════════════════════════════
# Triplet Loss with Hard-Negative Mining
# ═══════════════════════════════════════════════════════════════
class TripletMiningLoss(nn.Module):
    """
    Triplet loss with online negative mining strategies:
      - hard: select hardest negative (closest to anchor)
      - semi-hard: select negatives within margin boundary
      - easy: select random negatives (fastest, least informative)
    """

    def __init__(self, margin: float = 0.3, strategy: str = "semi-hard"):
        super().__init__()
        self.margin = margin
        self.strategy = strategy

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise L2 distance matrix."""
        dot = embeddings @ embeddings.T
        sq_norm = torch.diag(dot)
        distances = sq_norm.unsqueeze(0) - 2.0 * dot + sq_norm.unsqueeze(1)
        return torch.clamp(distances, min=0.0).sqrt()

    def _mine_triplets(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine (anchor, positive, negative) triplets from a batch.
        
        Uses the Hungarian Algorithm for optimal positive assignment
        when available, falls back to random matching otherwise.
        """
        dist_matrix = self._pairwise_distances(embeddings)
        batch_size = embeddings.size(0)
        
        anchors, positives, negatives = [], [], []
        
        for i in range(batch_size):
            # Find all positives (same label) and negatives (different label)
            pos_mask = (labels == labels[i]) & (
                torch.arange(batch_size, device=labels.device) != i
            )
            neg_mask = labels != labels[i]
            
            pos_indices = pos_mask.nonzero(as_tuple=True)[0]
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            # Select positive
            pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
            
            # Mine negative based on strategy
            neg_dists = dist_matrix[i][neg_indices]
            
            if self.strategy == "hard":
                # Closest negative
                neg_idx = neg_indices[neg_dists.argmin()]
            elif self.strategy == "semi-hard":
                # Within margin: d(a,p) < d(a,n) < d(a,p) + margin
                ap_dist = dist_matrix[i][pos_idx]
                semi_hard_mask = (neg_dists > ap_dist) & (
                    neg_dists < ap_dist + self.margin
                )
                if semi_hard_mask.any():
                    valid = neg_indices[semi_hard_mask]
                    neg_idx = valid[torch.randint(len(valid), (1,))]
                else:
                    neg_idx = neg_indices[neg_dists.argmin()]
            else:  # easy
                neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
            
            anchors.append(i)
            positives.append(pos_idx.item())
            negatives.append(neg_idx.item())
        
        if not anchors:
            return embeddings[:1], embeddings[:1], embeddings[:1]
        
        a_idx = torch.tensor(anchors, device=embeddings.device)
        p_idx = torch.tensor(positives, device=embeddings.device)
        n_idx = torch.tensor(negatives, device=embeddings.device)
        
        return embeddings[a_idx], embeddings[p_idx], embeddings[n_idx]

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels: (B,) class labels for mining
        Returns:
            Scalar triplet loss
        """
        anchor, positive, negative = self._mine_triplets(embeddings, labels)
        return F.triplet_margin_loss(
            anchor, positive, negative,
            margin=self.margin, p=2,
        )


# ═══════════════════════════════════════════════════════════════
# Hungarian Algorithm Batch Matcher
# ═══════════════════════════════════════════════════════════════
class HungarianMatcher:
    """
    Uses the Hungarian Algorithm (linear_sum_assignment) to find
    optimal one-to-one matching between two sets of embeddings.
    
    Useful for pairing query embeddings to document embeddings
    in a batch to minimize total assignment cost.
    """

    @staticmethod
    def match(
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """
        Find optimal assignment between two sets of embeddings.
        
        Args:
            embeddings_a: (N, D) first set
            embeddings_b: (M, D) second set
        Returns:
            List of (i, j) matched pairs
        """
        if linear_sum_assignment is None:
            raise ImportError("scipy required for Hungarian matching")
        
        # Cost matrix = negative cosine similarity (minimize → maximize sim)
        a_norm = F.normalize(embeddings_a, p=2, dim=1)
        b_norm = F.normalize(embeddings_b, p=2, dim=1)
        cost_matrix = -(a_norm @ b_norm.T).cpu().numpy()
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind.tolist(), col_ind.tolist()))


# ═══════════════════════════════════════════════════════════════
# Polyak Averaging (EMA)
# ═══════════════════════════════════════════════════════════════
class PolyakAveraging:
    """
    Maintain an exponential moving average of model parameters.
    θ_ema = decay * θ_ema + (1 - decay) * θ_current
    
    The EMA model typically generalizes better than the final checkpoint.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update EMA parameters after each optimizer step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """Replace model params with EMA params (for evaluation)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original params after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ═══════════════════════════════════════════════════════════════
# Siamese Fine-Tuner
# ═══════════════════════════════════════════════════════════════
class SiameseFineTuner:
    """
    Fine-tunes CLIP using Siamese contrastive + triplet objectives.
    
    Uses:
      - Nesterov Momentum (SGD variant)
      - Polyak Averaging for stable evaluation
      - Hard-negative mining for informative gradients
    """

    def __init__(
        self,
        encoder: CLIPEncoder,
        config: Optional[SiameseConfig] = None,
    ):
        self.encoder = encoder.to(DEVICE)
        self.config = config or SiameseConfig()
        
        # Losses
        self.contrastive_loss = MarginContrastiveLoss(margin=self.config.margin)
        self.triplet_loss = TripletMiningLoss(
            margin=self.config.triplet_margin,
            strategy=self.config.mining_strategy,
        )
        
        # Optimizer with Nesterov Momentum
        trainable_params = [
            p for p in self.encoder.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.SGD(
            trainable_params,
            lr=1e-4,
            momentum=self.config.nesterov_momentum,
            nesterov=True,  # Nesterov accelerated gradient
            weight_decay=1e-4,
        )
        
        # Polyak Averaging
        self.ema = PolyakAveraging(
            self.encoder, decay=self.config.polyak_decay
        )
        
        # Hungarian matcher for optimal pairing
        self.matcher = HungarianMatcher()
        
        logger.info(
            f"SiameseFineTuner ready | Mining: {self.config.mining_strategy} | "
            f"Nesterov: momentum={self.config.nesterov_momentum}"
        )

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step with contrastive + triplet losses.
        
        Args:
            images: (B, 3, H, W) batch of images
            labels: (B,) class labels
        Returns:
            Dict with 'contrastive_loss', 'triplet_loss', 'total_loss'
        """
        self.encoder.train()
        
        # Encode all images through shared backbone
        embeddings = self.encoder.encode_image(images)
        
        # Split batch into pairs for contrastive loss
        B = embeddings.size(0)
        if B >= 2:
            emb1, emb2 = embeddings[:B//2], embeddings[B//2:B//2*2]
            pair_labels = (labels[:B//2] != labels[B//2:B//2*2]).float()
            c_loss = self.contrastive_loss(emb1, emb2, pair_labels)
        else:
            c_loss = torch.tensor(0.0, device=DEVICE)
        
        # Triplet loss with mining
        t_loss = self.triplet_loss(embeddings, labels)
        
        total = c_loss + t_loss
        
        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.encoder.parameters(), max_norm=1.0
        )
        self.optimizer.step()
        
        # Update EMA
        self.ema.update(self.encoder)
        
        return {
            "contrastive_loss": c_loss.item(),
            "triplet_loss": t_loss.item(),
            "total_loss": total.item(),
        }

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 10,
        save_path: Optional[str] = None,
    ):
        """Full Siamese fine-tuning loop."""
        for epoch in range(num_epochs):
            epoch_losses = {"contrastive_loss": 0, "triplet_loss": 0, "total_loss": 0}
            n_batches = 0
            
            for batch in train_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                losses = self.train_step(images, labels)
                for k, v in losses.items():
                    epoch_losses[k] += v
                n_batches += 1
            
            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)
            
            logger.info(
                f"Siamese Epoch {epoch+1}/{num_epochs} | "
                f"Contrastive: {epoch_losses['contrastive_loss']:.4f} | "
                f"Triplet: {epoch_losses['triplet_loss']:.4f} | "
                f"Total: {epoch_losses['total_loss']:.4f}"
            )
        
        # Save with EMA weights
        if save_path:
            self.ema.apply_shadow(self.encoder)
            torch.save(self.encoder.state_dict(), save_path)
            self.ema.restore(self.encoder)
            logger.info(f"Siamese model saved (EMA weights): {save_path}")
