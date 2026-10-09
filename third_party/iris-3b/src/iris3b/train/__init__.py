"""Training stack: optimizer, LR schedules, EMA, checkpointing, orchestration."""

from iris3b.train.ckpt import load_checkpoint, resolve_resume, save_checkpoint
from iris3b.train.ema import EMA
from iris3b.train.lr import build_lr_scheduler, constant_with_warmup, cosine_with_warmup
from iris3b.train.optim import build_optimizer
from iris3b.train.trainer import Trainer, TrainModel, seed_everything

__all__ = [
    "EMA",
    "TrainModel",
    "Trainer",
    "build_lr_scheduler",
    "build_optimizer",
    "constant_with_warmup",
    "cosine_with_warmup",
    "load_checkpoint",
    "resolve_resume",
    "save_checkpoint",
    "seed_everything",
]
