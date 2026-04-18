"""
Optuna Hyperparameter Tuning for CLIP Fine-Tuning
==================================================
Techniques implemented:
  - Optuna with TPE (Tree-structured Parzen Estimator) sampler
  - MedianPruner for early stopping of unpromising trials
  - Bayesian optimization over: LR, weight decay, LoRA rank,
    batch size, contrastive temperature, ArcFace margin/scale
  - Adagrad, Adadelta, BFGS optimizer comparison
  - Learning Rate Decay schedules comparison
"""

import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
except ImportError:
    optuna = None
    logger.warning("optuna not installed. Run: pip install optuna")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_config, DEVICE, CLIPConfig, OptunaConfig
from src.embeddings.clip_encoder import CLIPEncoder, CLIPTrainer, ContrastiveLoss


# ═══════════════════════════════════════════════════════════════
# Optimizer Factory (AdamW, Adagrad, Adadelta, BFGS)
# ═══════════════════════════════════════════════════════════════
def create_optimizer(
    params,
    name: str,
    lr: float,
    weight_decay: float = 0.01,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """
    Create optimizer by name. Supports:
      - adamw: AdamW with weight decay decoupling
      - adagrad: Adaptive per-parameter learning rates
      - adadelta: No manual LR needed (adaptive)
      - sgd_nesterov: SGD with Nesterov momentum
      - lbfgs: Limited-memory BFGS (quasi-Newton)
    """
    name = name.lower()
    
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif name == "adadelta":
        return torch.optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_nesterov":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum,
            nesterov=True, weight_decay=weight_decay,
        )
    elif name == "lbfgs":
        return torch.optim.LBFGS(params, lr=lr, max_iter=20)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ═══════════════════════════════════════════════════════════════
# LR Scheduler Factory
# ═══════════════════════════════════════════════════════════════
def create_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    num_epochs: int,
    **kwargs,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create LR scheduler by name. Supports:
      - cosine_annealing: Cosine Annealing with Warm Restarts
      - step_decay: StepLR (learning rate decay)
      - exponential: Exponential decay
      - reduce_on_plateau: ReduceLROnPlateau
      - one_cycle: OneCycleLR
    """
    name = name.lower()
    
    if name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=kwargs.get("T_0", 5), T_mult=kwargs.get("T_mult", 2),
        )
    elif name == "step_decay":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get("step_size", 5), gamma=0.5,
        )
    elif name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs.get("gamma", 0.95),
        )
    elif name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )
    elif name == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 1e-3),
            epochs=num_epochs,
            steps_per_epoch=kwargs.get("steps_per_epoch", 100),
        )
    else:
        return None


# ═══════════════════════════════════════════════════════════════
# Optuna Objective Function
# ═══════════════════════════════════════════════════════════════
def create_objective(
    train_dataset,
    val_dataset,
    num_classes: int,
    max_epochs: int = 5,
    optuna_config: Optional[OptunaConfig] = None,
):
    """
    Factory that creates an Optuna objective closure.
    
    Tunes:
      - learning_rate
      - weight_decay
      - lora_rank
      - batch_size
      - contrastive_temperature
      - arcface_scale, arcface_margin
      - optimizer type
      - lr_scheduler type
      - gradient_accumulation_steps
    """
    cfg = optuna_config or OptunaConfig()
    
    def objective(trial: "optuna.Trial") -> float:
        """Single trial: build model, train briefly, return val loss."""
        
        # ── Sample hyperparameters ──
        lr = trial.suggest_float(
            "learning_rate", *cfg.lr_range, log=True
        )
        weight_decay = trial.suggest_float(
            "weight_decay", *cfg.weight_decay_range, log=True
        )
        lora_rank = trial.suggest_int(
            "lora_rank", *cfg.lora_rank_range, step=4
        )
        batch_size = trial.suggest_categorical(
            "batch_size", cfg.batch_size_choices
        )
        temperature = trial.suggest_float(
            "temperature", *cfg.temperature_range
        )
        arcface_scale = trial.suggest_float(
            "arcface_scale", 10.0, 64.0
        )
        arcface_margin = trial.suggest_float(
            "arcface_margin", 0.1, 0.8
        )
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["adamw", "adagrad", "adadelta", "sgd_nesterov"]
        )
        scheduler_name = trial.suggest_categorical(
            "lr_scheduler", [
                "cosine_annealing", "step_decay", "exponential", "one_cycle"
            ]
        )
        grad_accum = trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 4, 8]
        )
        label_smoothing = trial.suggest_float(
            "label_smoothing", 0.0, 0.2
        )
        
        # ── Build config ──
        config = CLIPConfig(
            learning_rate=lr,
            weight_decay=weight_decay,
            lora_rank=lora_rank,
            batch_size=batch_size,
            contrastive_temperature=temperature,
            arcface_scale=arcface_scale,
            arcface_margin=arcface_margin,
            gradient_accumulation_steps=grad_accum,
            label_smoothing=label_smoothing,
            num_epochs=max_epochs,
            lora_enabled=True,
            swa_enabled=False,  # Disable SWA during tuning for speed
        )
        
        # ── Build model ──
        encoder = CLIPEncoder(config)
        encoder.apply_lora()
        
        # ── Data loaders ──
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )
        
        # ── Custom training loop for Optuna ──
        encoder = encoder.to(DEVICE)
        contrastive_fn = ContrastiveLoss(
            temperature=temperature,
            label_smoothing=label_smoothing,
        )
        
        trainable_params = [
            p for p in encoder.parameters() if p.requires_grad
        ]
        optimizer = create_optimizer(
            trainable_params, optimizer_name, lr, weight_decay
        )
        scheduler = create_scheduler(
            optimizer, scheduler_name, max_epochs,
            steps_per_epoch=len(train_loader),
        )
        
        # ── Train ──
        for epoch in range(max_epochs):
            encoder.train()
            epoch_loss = 0.0
            n = 0
            
            for batch in train_loader:
                images = batch["image"].to(DEVICE)
                texts = batch["text"].to(DEVICE)
                
                outputs = encoder(images=images, texts=texts)
                loss = contrastive_fn(
                    outputs["image_embeddings"],
                    outputs["text_embeddings"],
                )
                loss = loss / grad_accum
                loss.backward()
                
                if (n + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * grad_accum
                n += 1
            
            # Step scheduler
            if scheduler is not None:
                if scheduler_name == "reduce_on_plateau":
                    scheduler.step(epoch_loss / max(n, 1))
                elif scheduler_name != "one_cycle":
                    scheduler.step()
            
            # ── Validate ──
            encoder.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(DEVICE)
                    texts = batch["text"].to(DEVICE)
                    outputs = encoder(images=images, texts=texts)
                    loss = contrastive_fn(
                        outputs["image_embeddings"],
                        outputs["text_embeddings"],
                    )
                    val_loss += loss.item()
                    val_n += 1
            
            avg_val_loss = val_loss / max(val_n, 1)
            
            # Report to Optuna for pruning
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Clean up GPU memory
        del encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return avg_val_loss
    
    return objective


# ═══════════════════════════════════════════════════════════════
# Run Hyperparameter Search
# ═══════════════════════════════════════════════════════════════
def run_hyperparameter_search(
    train_dataset,
    val_dataset,
    num_classes: int = 1000,
    config: Optional[OptunaConfig] = None,
    study_name: str = "clip_finetuning",
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Launch Optuna hyperparameter search.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        num_classes: Number of classes for ArcFace
        config: Optuna configuration
        study_name: Name for the study (for persistence)
        storage: Optuna storage URL (sqlite for persistence)
    
    Returns:
        Dict with best params and study statistics
    """
    if optuna is None:
        raise ImportError("optuna required: pip install optuna")
    
    cfg = config or OptunaConfig()
    
    # Select sampler
    sampler_map = {
        "tpe": TPESampler(seed=42),
        "random": RandomSampler(seed=42),
        "cmaes": CmaEsSampler(seed=42),
    }
    sampler = sampler_map.get(cfg.sampler, TPESampler(seed=42))
    
    # Select pruner
    pruner_map = {
        "median": MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        "hyperband": HyperbandPruner(min_resource=1, max_resource=10),
    }
    pruner = pruner_map.get(cfg.pruner, MedianPruner())
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    
    # Create objective
    objective = create_objective(
        train_dataset, val_dataset, num_classes,
        max_epochs=5,
        optuna_config=cfg,
    )
    
    # Run optimization
    logger.info(
        f"Starting Optuna search: {cfg.n_trials} trials | "
        f"Sampler: {cfg.sampler} | Pruner: {cfg.pruner}"
    )
    
    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        timeout=cfg.timeout_seconds,
        show_progress_bar=True,
    )
    
    # Results
    best = study.best_trial
    logger.info(f"Best trial #{best.number}: val_loss={best.value:.4f}")
    logger.info(f"Best params: {best.params}")
    
    return {
        "best_params": best.params,
        "best_value": best.value,
        "best_trial_number": best.number,
        "n_trials_completed": len(study.trials),
        "n_pruned": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]),
    }
