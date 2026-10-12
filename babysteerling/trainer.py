import math
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_concepts.nn import InferenceOutput

from .backbone.lm import LM


class LightningLM(pl.LightningModule):
    """PyTorch Lightning wrapper for LM models (supports both Causal and Diffusion).

    Handles training steps, validation evaluation, metric logging, and LR scheduling.
    """

    def __init__(
        self,
        model: LM,
        lr: float = 6e-4,
        min_lr: float = 6e-5,
        warmup_steps: int = 2000,
        max_steps: int = 50000,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, *args, **kwargs) -> InferenceOutput:
        return self.model(*args, **kwargs)

    def _total_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return sum(losses.values())

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses = self.model.step(batch, step=self.global_step)
        total_loss = self._total_loss(losses)

        self.log(
            "train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        for name, val in losses.items():
            self.log(
                f"train/{name}_loss", val, on_step=True, on_epoch=False, sync_dist=True
            )

        return total_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        losses = self.model.step(batch, step=self.global_step)
        total_loss = self._total_loss(losses)

        self.log(
            "val/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for name, val in losses.items():
            self.log(
                f"val/{name}_loss", val, on_step=False, on_epoch=True, sync_dist=True
            )

    def on_validation_epoch_end(self) -> None:
        """Computes and logs accumulated epoch-level evaluation metrics (e.g., ConceptAUC)."""
        for name, metric in self.model.metrics.items():
            value = metric.compute()
            if value is None:
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    self.log(
                        f"val/{name}_{k}",
                        v,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )
            else:
                self.log(
                    f"val/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            metric.reset()

    def configure_optimizers(self):
        # Separate parameters that decay vs weight decay exclusions (biases, layernorms, 1D terms)
        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Exclude 1D tensors (biases, layernorms) and specific embedding parameters from decay
            if param.ndim < 2 or any(
                nd_name in name.lower()
                for nd_name in ["bias", "norm", "ln_", "embedding"]
            ):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(optim_groups, lr=self.hparams.lr, betas=self.hparams.betas)

        # Cosine learning rate scheduler with linear warmup
        def lr_lambda(current_step: int) -> float:
            if current_step < self.hparams.warmup_steps:
                return float(current_step + 1) / float(
                    max(1, self.hparams.warmup_steps)
                )

            progress = float(current_step - self.hparams.warmup_steps) / float(
                max(1, self.hparams.max_steps - self.hparams.warmup_steps)
            )
            # Clamp progress to 1.0 to ensure decay stays at min_lr after max_steps
            progress = min(1.0, max(0.0, progress))

            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = self.hparams.min_lr / self.hparams.lr
            return min_ratio + (1.0 - min_ratio) * factor

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Step per batch iteration
                "frequency": 1,
            },
        }
