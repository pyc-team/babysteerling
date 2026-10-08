from pytorch_lightning.callbacks import Callback
import torch
import wandb


class TextGenerationLogger(Callback):
    """Generates and logs a decoded text sample to WandB when trainer.fit() finishes."""

    def __init__(self, decode_fn, **gen_kwargs):
        super().__init__()
        self.decode_fn = decode_fn
        self.gen_kwargs = gen_kwargs

    @torch.no_grad()
    def on_fit_end(self, trainer, pl_module) -> None:
        model = pl_module.model
        model.eval()

        # Unpacks whatever generation parameters were passed at instantiation
        sample_ids = model.generate(batch_size=1, **self.gen_kwargs)

        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids[0].tolist()

        sample_text = self.decode_fn(sample_ids)

        print("\n" + "=" * 50)
        print(
            f"[Fit Complete | Step {trainer.global_step}] Generated Text Sample:\n{sample_text}"
        )
        print("=" * 50 + "\n")

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {
                    "samples/final_generated_text": wandb.Html(
                        f"<pre style='white-space: pre-wrap; font-family: monospace;'>{sample_text}</pre>"
                    ),
                    "global_step": trainer.global_step,
                }
            )


class DiagnosticsLogger(Callback):
    """Calls model.diagnostics() and logs the dict it returns -- appended to the same W&B run,
    not a new one. Also callable directly (log_diagnostics), for runs that load an
    already-trained checkpoint instead of calling trainer.fit().
    """

    def __init__(self, decode_fn):
        super().__init__()
        self.decode_fn = decode_fn

    @torch.no_grad()
    def on_fit_end(self, trainer, pl_module) -> None:
        self.log_diagnostics(trainer, pl_module)

    @torch.no_grad()
    def log_diagnostics(self, trainer, pl_module) -> None:
        model = pl_module.model
        model.eval()
        diagnostics = model.diagnostics(self.decode_fn)
        if not diagnostics:
            return

        print(f"\nDiagnostics (step {trainer.global_step}): {diagnostics}")

        if trainer.logger and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {f"diagnostics/{k}": v for k, v in diagnostics.items()}
            )
