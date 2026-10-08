import csv
import hashlib
import json
import os

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger

import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from torch_concepts.nn import DeterministicInference

from babysteerling import diffusion
from babysteerling.callback import DiagnosticsLogger
from babysteerling.data.utils import load_tokenizer
from babysteerling.loader import ConceptDataModule
from babysteerling.trainer import LightningLM


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    # 1. Device resolution
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    if (
        device == "mps"
        and getattr(cfg.model, "known_encoder_type", None) == "product_key"
    ):
        print(
            "known_encoder_type='product_key': forcing device='cpu' (MPS crash fix needs small Kt)"
        )
        device = "cpu"

    # 2. Setup logging and checkpointing
    # Compute a hash of the config (excluding wandb) to uniquely identify this run's configuration
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    hash_payload = {k: v for k, v in config_dict.items() if k != "wandb"}
    config_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    # Checkpoint directory keyed strictly by config_hash
    run_ckpt_dir = os.path.join("./checkpoints", cfg.wandb.project, config_hash)
    resume_ckpt_path = os.path.join(run_ckpt_dir, "last.ckpt")
    ckpt_exists = os.path.exists(resume_ckpt_path)
    # W&B logger setup
    choices = HydraConfig.get().runtime.choices
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        name=choices.get("variant") or getattr(cfg.model, "wandb_name", None),
        mode=cfg.wandb.mode,
        config=config_dict,
        save_dir="./logs",
    )

    # 3. Load data and tokenizer
    tok, vocab_size, decode = load_tokenizer(cfg.data.data_dir)
    dm = ConceptDataModule(
        data_dir=cfg.data.data_dir,
        block_size=cfg.data.block_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        min_lifted_tokens=cfg.data.min_lifted_tokens,
        pin_memory=cfg.data.pin_memory,
        train_val_split=cfg.data.train_val_split,
    )
    dm.setup("fit")

    # 4. Dynamic model (architecture + loss) instantiation
    mask_token_id = diffusion.ensure_mask_token(tok)
    loss_fn = instantiate(cfg.loss) if "loss" in cfg else None
    model_kwargs = {
        "vocab_size": len(tok.get_vocab()),
        "block_size": cfg.data.block_size,
        "loss_fn": loss_fn,
        "inference": DeterministicInference,
        # the following are used only by some model types and ignored by others
        "out_concepts": [c["label"] for c in dm.concepts],
        "mask_token_id": mask_token_id,
        "steering_every_n_steps": cfg.steering.every_n_steps,
        "inj_layer": cfg.steering.inj_layer,
        "tau": cfg.steering.tau,
    }
    model = instantiate(cfg.model, **model_kwargs, _recursive_=False)

    # 5. Wrap in PyTorch Lightning Module
    pl_model = LightningLM(
        model=model,
        lr=cfg.training.lr,
        min_lr=cfg.training.min_lr,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
    )

    # 6. Instantiate Trainer
    gen_logger = instantiate(cfg.training.generation_callback, decode_fn=decode)
    diagnostics_logger = DiagnosticsLogger(decode_fn=decode)
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_ckpt_dir,
        filename=cfg.training.checkpoint_filename,
        save_last=cfg.training.save_last,
        every_n_train_steps=cfg.training.every_n_train_steps,
    )
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_progress_bar=cfg.training.enable_progress_bar,
        enable_checkpointing=cfg.training.enable_checkpointing,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        limit_val_batches=cfg.training.limit_val_batches,
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            checkpoint_callback,
            gen_logger,
            diagnostics_logger,
        ],
    )

    # 7. Start Training
    try:
        if ckpt_exists:
            if cfg.training.resume_from_checkpoint:
                print(
                    f"\n[Resumption] Found existing checkpoint for hash '{config_hash}'. Resuming from: {resume_ckpt_path}\n"
                )
                trainer.fit(model=pl_model, datamodule=dm, ckpt_path=resume_ckpt_path)
            else:
                print(
                    f"\n[Already Trained] Loading checkpoint for hash '{config_hash}' without retraining: {resume_ckpt_path}\n"
                )
                pl_model = LightningLM.load_from_checkpoint(
                    resume_ckpt_path, model=model
                )
                diagnostics_logger.log_diagnostics(trainer, pl_model)
        else:
            print(
                f"\n[Fresh Run] No checkpoint found for hash '{config_hash}'. Starting new run...\n"
            )
            trainer.fit(model=pl_model, datamodule=dm)
        print("\nTrainer execution finished successfully!")

        # 8. Report: validation metrics -> CSV, alongside the resolved config,
        # both keyed by config_hash so they sit next to the checkpoint they belong to.
        config_path = os.path.join(run_ckpt_dir, "config.yaml")
        if not os.path.exists(config_path):
            OmegaConf.save(cfg, config_path)

        val_results = trainer.validate(model=pl_model, datamodule=dm)
        metrics = val_results[0] if val_results else {}
        metrics_path = os.path.join(run_ckpt_dir, "metrics.csv")
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
        print(f"\n[Report] Wrote validation metrics to: {metrics_path}\n")

        # Concept top-k heatmap data (only models overriding LM.diagnostics() return this,
        # e.g. ConceptResidualALM -- no-op dict for every other model type).
        pl_model.model.eval()
        with torch.no_grad():
            diagnostics = pl_model.model.diagnostics(decode)
        heatmap_fig = diagnostics.get("concept_topk_heatmap")
        if heatmap_fig is not None:
            trace = heatmap_fig.data[0]
            heatmap_path = os.path.join(run_ckpt_dir, "concept_topk_heatmap.json")
            with open(heatmap_path, "w") as f:
                json.dump(
                    {
                        "concepts": [str(c) for c in trace.y],
                        "columns": [str(c) for c in trace.x],
                        "weights": [[float(w) for w in row] for row in trace.z],
                        "tokens": [[str(t) for t in row] for row in trace.text],
                    },
                    f,
                )
            print(f"[Report] Wrote concept top-k heatmap data to: {heatmap_path}\n")
    finally:
        wandb.finish()

    print("\nTrainer execution finished successfully!")


if __name__ == "__main__":
    main()
