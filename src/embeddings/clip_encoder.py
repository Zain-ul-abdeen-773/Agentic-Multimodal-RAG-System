"""
CLIP Encoder with Advanced Fine-Tuning Pipeline
================================================
Techniques implemented:
  - OpenCLIP ViT-B/32 dual encoder
  - ArcFace loss head for discriminative embeddings
  - InfoNCE contrastive loss for cross-modal alignment
  - LoRA (Low-Rank Adaptation) for parameter-efficient tuning
  - Cosine Annealing LR with Warm Restarts
  - AdamW optimizer with configurable weight decay
  - Gradient Accumulation for effective large-batch training
  - Gradient Clipping (max_norm)
  - Mixed Precision Training (torch.cuda.amp)
  - Kaiming / Xavier weight initialization
  - Label Smoothing on classification objectives
  - Stochastic Weight Averaging (SWA)
  - Multi-head Attention projection layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

try:
    import open_clip
except ImportError:
    open_clip = None
    logger.warning("open_clip not installed. Run: pip install open-clip-torch")

try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    get_peft_model, LoraConfig, TaskType = None, None, None
    logger.warning("peft not installed. LoRA disabled. Run: pip install peft")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, DEVICE, CLIPConfig


# ═══════════════════════════════════════════════════════════════
# ArcFace Loss Head
# ═══════════════════════════════════════════════════════════════
class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) for discriminative embeddings.
    
    Pushes embeddings of same class closer on the hypersphere while
    maintaining angular margin between different classes.
    
    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss
    for Deep Face Recognition", CVPR 2019.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 1000,
        scale: float = 30.0,
        margin: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.label_smoothing = label_smoothing
        
        # Learnable class centers on unit hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold to prevent cos(theta + m) from decreasing when theta > pi - m
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels: (B,) integer class labels
        Returns:
            Scaled logits with angular margin applied to ground-truth class
        """
        # Normalize both embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity → cos(θ)
        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, max=1.0))
        
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Numerical stability: if cos(θ) < threshold, use linearized version
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot for target class → apply margin only to ground-truth
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
        
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
        
        return logits


# ═══════════════════════════════════════════════════════════════
# Contrastive Loss (InfoNCE / CLIP-style)
# ═══════════════════════════════════════════════════════════════
class ContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE loss for cross-modal (image ↔ text) alignment.
    This is the standard CLIP training objective.
    """

    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_embeddings: (B, D) normalized image features
            text_embeddings: (B, D) normalized text features
        Returns:
            Scalar loss (average of image→text and text→image)
        """
        # Similarity matrix
        logits = (image_embeddings @ text_embeddings.T) / self.temperature
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric cross-entropy with optional label smoothing
        loss_i2t = F.cross_entropy(
            logits, labels, label_smoothing=self.label_smoothing
        )
        loss_t2i = F.cross_entropy(
            logits.T, labels, label_smoothing=self.label_smoothing
        )
        
        return (loss_i2t + loss_t2i) / 2.0


# ═══════════════════════════════════════════════════════════════
# Projection Head with Multi-Head Attention
# ═══════════════════════════════════════════════════════════════
class ProjectionHead(nn.Module):
    """
    Non-linear projection head applied on top of CLIP embeddings.
    Uses Kaiming/Xavier initialization and optional multi-head attention
    for richer representations.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_heads: int = 4,
        init_method: str = "kaiming",
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Apply weight initialization
        self._init_weights(init_method)

    def _init_weights(self, method: str):
        """Kaiming or Xavier initialization for linear layers."""
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif method == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) input embeddings
        Returns:
            (B, output_dim) projected and normalized embeddings
        """
        # Self-attention over embedding (treat as sequence of length 1)
        x_unsq = x.unsqueeze(1)  # (B, 1, D)
        attn_out, _ = self.attention(x_unsq, x_unsq, x_unsq)
        x = self.layer_norm(x + attn_out.squeeze(1))
        
        # Project and L2-normalize
        projected = self.projector(x)
        return F.normalize(projected, p=2, dim=-1)


# ═══════════════════════════════════════════════════════════════
# Main CLIP Encoder Module
# ═══════════════════════════════════════════════════════════════
class CLIPEncoder(nn.Module):
    """
    Full CLIP encoder with:
      - OpenCLIP backbone (frozen or LoRA-adapted)
      - Projection heads for image and text
      - ArcFace classification head (optional)
    """

    def __init__(self, config: Optional[CLIPConfig] = None):
        super().__init__()
        self.config = config or CLIPConfig()
        
        # Load OpenCLIP model
        if open_clip is None:
            raise ImportError("open_clip required: pip install open-clip-torch")
        
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=self.config.pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(self.config.model_name)
        
        # Freeze CLIP backbone by default (LoRA will add trainable params)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Image and text projection heads
        self.image_projection = ProjectionHead(
            input_dim=self.config.embedding_dim,
            output_dim=self.config.embedding_dim,
            init_method=self.config.init_method,
        )
        self.text_projection = ProjectionHead(
            input_dim=self.config.embedding_dim,
            output_dim=self.config.embedding_dim,
            init_method=self.config.init_method,
        )
        
        logger.info(
            f"CLIPEncoder initialized: {self.config.model_name} | "
            f"LoRA={self.config.lora_enabled} | Device={DEVICE}"
        )

    def apply_lora(self):
        """
        Apply LoRA adapters to the CLIP visual transformer.
        Only adds trainable low-rank matrices to attention projections,
        keeping >95% of parameters frozen.
        """
        if LoraConfig is None:
            logger.warning("peft not available, skipping LoRA")
            return
        
        if not self.config.lora_enabled:
            return
        
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )
        
        # Apply LoRA to visual encoder
        self.clip_model.visual = get_peft_model(
            self.clip_model.visual, lora_config
        )
        
        trainable = sum(
            p.numel() for p in self.clip_model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.clip_model.parameters())
        logger.info(
            f"LoRA applied: {trainable:,} trainable / {total:,} total "
            f"({100*trainable/total:.2f}%)"
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images through CLIP backbone + projection.
        
        Args:
            images: (B, 3, H, W) preprocessed image tensor
        Returns:
            (B, D) L2-normalized image embeddings
        """
        with torch.no_grad() if not self.config.lora_enabled else torch.enable_grad():
            features = self.clip_model.encode_image(images)
        
        features = features.float()
        projected = self.image_projection(features)
        return projected

    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """
        Encode tokenized text through CLIP backbone + projection.
        
        Args:
            texts: (B, seq_len) tokenized text tensor
        Returns:
            (B, D) L2-normalized text embeddings
        """
        with torch.no_grad():
            features = self.clip_model.encode_text(texts)
        
        features = features.float()
        projected = self.text_projection(features)
        return projected

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        texts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning embeddings for available modalities.
        
        Returns:
            Dict with 'image_embeddings' and/or 'text_embeddings'
        """
        outputs = {}
        if images is not None:
            outputs["image_embeddings"] = self.encode_image(images)
        if texts is not None:
            outputs["text_embeddings"] = self.encode_text(texts)
        return outputs

    def tokenize(self, texts: List[str]) -> torch.Tensor:
        """Tokenize a list of strings using the CLIP tokenizer."""
        return self.tokenizer(texts)


# ═══════════════════════════════════════════════════════════════
# Training Pipeline
# ═══════════════════════════════════════════════════════════════
class CLIPTrainer:
    """
    Full training pipeline with:
      - Contrastive + ArcFace losses
      - AdamW + Cosine Annealing LR with warm restarts
      - Gradient accumulation & clipping
      - Mixed precision (AMP)
      - SWA (Stochastic Weight Averaging)
      - Comprehensive logging
    """

    def __init__(
        self,
        model: CLIPEncoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[CLIPConfig] = None,
        num_classes: int = 1000,
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or CLIPConfig()
        
        # Losses
        self.contrastive_loss = ContrastiveLoss(
            temperature=self.config.contrastive_temperature,
            label_smoothing=self.config.label_smoothing,
        )
        self.arcface_head = ArcFaceHead(
            embedding_dim=self.config.embedding_dim,
            num_classes=num_classes,
            scale=self.config.arcface_scale,
            margin=self.config.arcface_margin,
            label_smoothing=self.config.label_smoothing,
        ).to(DEVICE)
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
        
        # Optimizer — AdamW with weight decay
        trainable_params = [
            {"params": model.image_projection.parameters(), "lr": self.config.learning_rate},
            {"params": model.text_projection.parameters(), "lr": self.config.learning_rate},
            {"params": self.arcface_head.parameters(), "lr": self.config.learning_rate * 10},
        ]
        # Add LoRA params if present
        lora_params = [
            p for p in model.clip_model.parameters() if p.requires_grad
        ]
        if lora_params:
            trainable_params.append(
                {"params": lora_params, "lr": self.config.learning_rate * 0.1}
            )
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # LR Scheduler — Cosine Annealing with Warm Restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.cosine_T_0,
            T_mult=self.config.cosine_T_mult,
            eta_min=self.config.cosine_eta_min,
        )
        
        # Mixed precision
        self.scaler = GradScaler(enabled=self.config.mixed_precision)
        
        # SWA
        self.swa_model = None
        self.swa_scheduler = None
        if self.config.swa_enabled:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(
                self.optimizer,
                swa_lr=self.config.swa_lr,
            )
        
        # Metrics tracking
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch with gradient accumulation and AMP."""
        self.model.train()
        self.arcface_head.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(DEVICE)
            texts = batch["text"].to(DEVICE)
            labels = batch.get("label", None)
            if labels is not None:
                labels = labels.to(DEVICE)
            
            # Forward pass with AMP
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(images=images, texts=texts)
                img_emb = outputs["image_embeddings"]
                txt_emb = outputs["text_embeddings"]
                
                # Contrastive loss (CLIP-style)
                loss = self.contrastive_loss(img_emb, txt_emb)
                
                # ArcFace loss (if labels available)
                if labels is not None:
                    arc_logits = self.arcface_head(img_emb, labels)
                    arc_loss = self.ce_loss(arc_logits, labels)
                    loss = loss + 0.5 * arc_loss
                
                # Scale for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward with scaler
            self.scaler.scale(loss).backward()
            
            # Step optimizer every N accumulation steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
        
        # LR scheduling
        use_swa = (
            self.config.swa_enabled
            and epoch >= self.config.swa_start_epoch
        )
        if use_swa:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        else:
            self.scheduler.step(epoch)
        
        avg_loss = total_loss / max(num_batches, 1)
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.history["train_loss"].append(avg_loss)
        self.history["learning_rate"].append(current_lr)
        
        logger.info(
            f"Epoch {epoch+1}/{self.config.num_epochs} | "
            f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
            f"SWA: {'active' if use_swa else 'off'}"
        )
        
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and compute loss."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            images = batch["image"].to(DEVICE)
            texts = batch["text"].to(DEVICE)
            
            outputs = self.model(images=images, texts=texts)
            loss = self.contrastive_loss(
                outputs["image_embeddings"],
                outputs["text_embeddings"],
            )
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.history["val_loss"].append(avg_loss)
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, save_path: Optional[str] = None):
        """Full training loop across all epochs."""
        logger.info(
            f"Starting training: {self.config.num_epochs} epochs | "
            f"Batch: {self.config.batch_size} | "
            f"Effective: {self.config.batch_size * self.config.gradient_accumulation_steps} | "
            f"Device: {DEVICE}"
        )
        
        best_val_loss = float("inf")
        
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(save_path, epoch, val_loss)
        
        # Update batch normalization for SWA model
        if self.swa_model is not None:
            logger.info("Updating SWA batch norm statistics...")
            torch.optim.swa_utils.update_bn(
                self.train_loader, self.swa_model, device=DEVICE
            )
        
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint with full state."""
        save_dict = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "arcface_state_dict": self.arcface_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "history": self.history,
        }
        if self.swa_model is not None:
            save_dict["swa_state_dict"] = self.swa_model.state_dict()
        
        torch.save(save_dict, path)
        logger.info(f"Checkpoint saved: {path} (epoch {epoch+1})")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.arcface_head.load_state_dict(checkpoint["arcface_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint.get("history", self.history)
        logger.info(
            f"Checkpoint loaded: {path} (epoch {checkpoint['epoch']+1})"
        )


# ═══════════════════════════════════════════════════════════════
# Convenience: Embed images/text for indexing
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def embed_images(
    encoder: CLIPEncoder,
    image_paths: List[str],
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Batch-encode a list of image file paths into embeddings.
    
    Returns:
        (N, D) tensor of L2-normalized embeddings
    """
    from PIL import Image
    
    encoder.eval()
    all_embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                img = encoder.preprocess(img)
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")
                continue
        
        if images:
            batch_tensor = torch.stack(images).to(DEVICE)
            emb = encoder.encode_image(batch_tensor)
            all_embeddings.append(emb.cpu())
    
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)


@torch.no_grad()
def embed_texts(
    encoder: CLIPEncoder,
    texts: List[str],
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Batch-encode a list of text strings into embeddings.
    
    Returns:
        (N, D) tensor of L2-normalized embeddings
    """
    encoder.eval()
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = encoder.tokenize(batch_texts).to(DEVICE)
        emb = encoder.encode_text(tokens)
        all_embeddings.append(emb.cpu())
    
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)
