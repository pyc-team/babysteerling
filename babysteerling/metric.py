import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics import Metric
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision


class NullMetric(nn.Module):
    """No-op stand-in for an unconfigured metric term."""

    def update(self, **kwargs) -> None:
        return None

    def compute(self):
        return None

    def reset(self) -> None:
        return None


class TokenAccuracy(Metric):
    """Next-token prediction accuracy aggregating across batches."""

    def __init__(self, ignore_index: int = -100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

        # Register state variables for proper batch accumulation and DDP reduction
        self.add_state(
            "correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, prediction: torch.Tensor, target: torch.Tensor, *extra):
        if prediction is None:
            return

        # Flatten tensors if necessary: prediction -> [N, V], target -> [N]
        if prediction.ndim == 3:
            prediction = prediction.reshape(-1, prediction.size(-1))
        target = target.reshape(-1)

        # Mask out padding / ignore_index tokens (e.g. -100)
        mask = target != self.ignore_index
        if not mask.any():
            return

        pred_ids = prediction[mask].argmax(dim=-1)
        valid_targets = target[mask]

        self.correct += (pred_ids == valid_targets).sum()
        self.total += valid_targets.numel()

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.correct.float() / self.total).item()


class ConceptAUC(Metric):
    """Computes Macro & Micro ROC-AUC and PR-AUC for document-level soft-OR predictions."""

    def __init__(self):
        super().__init__()
        self.num_concepts: int | None = None
        self.auroc_metric: MultilabelAUROC | None = None
        self.ap_metric: MultilabelAveragePrecision | None = None
        self.reset()

    def _lazy_init(self, num_concepts: int, device: torch.device):
        """Initializes state and metrics once num_concepts is known at runtime."""
        self.num_concepts = num_concepts
        self.auroc_metric = MultilabelAUROC(num_labels=num_concepts, average=None).to(
            device
        )
        self.ap_metric = MultilabelAveragePrecision(
            num_labels=num_concepts, average=None
        ).to(device)

    def reset(self):
        self.all_preds = []
        self.all_targets = []
        if self.auroc_metric is not None:
            self.auroc_metric.reset()
        if self.ap_metric is not None:
            self.ap_metric.reset()

    def update(self, concept_logits: torch.Tensor, doc_spans: list, *extra):
        if concept_logits is None:
            return

        if not doc_spans:
            return

        # Dynamically determine num_concepts on first forward pass
        if self.num_concepts is None:
            num_concepts = concept_logits.shape[-1]
            self._lazy_init(num_concepts, device=concept_logits.device)

        for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
            # Span logits shape: [doc_len, n]
            z_span = concept_logits[batch_idx, tok_start:tok_end, :]

            # log P(absent) = sum over tokens of -softplus(z_i)
            log_p_none = -F.softplus(z_span).sum(dim=0)  # shape: [n]

            # P(any) = 1 - exp(log_p_none)
            p_any = -torch.expm1(log_p_none)  # shape: [n]

            targets = torch.zeros(
                self.num_concepts, dtype=torch.long, device=p_any.device
            )
            if len(concept_ids) > 0:
                targets[concept_ids] = 1

            self.all_preds.append(p_any)
            self.all_targets.append(targets)

    def compute(self) -> dict[str, float]:
        if not self.all_preds:
            return {"macro_roc_auc": 0.0, "macro_pr_auc": 0.0}

        preds = torch.stack(self.all_preds)  # shape: [num_docs, num_concepts]
        targets = torch.stack(self.all_targets)  # shape: [num_docs, num_concepts]

        # Per-concept AUCs (shape: [num_concepts])
        per_concept_roc_auc = self.auroc_metric(preds, targets)
        per_concept_pr_auc = self.ap_metric(preds, targets)

        return {
            "macro_roc_auc": per_concept_roc_auc.nanmean().item(),
            "macro_pr_auc": per_concept_pr_auc.nanmean().item(),
        }


class CausalConceptEffect(Metric):
    """Mean absolute difference between two predictions when a concept is forced on vs. forced off"""

    def __init__(self):
        super().__init__()
        self.add_state("total_effect", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "total_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, logits_1: torch.Tensor, logits_0: torch.Tensor):
        if logits_1 is None or logits_0 is None:
            return

        diff = (logits_1 - logits_0).abs()  # shape: [B, T, vocab]
        self.total_effect += diff.sum()
        self.total_count += diff.numel()

    def compute(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.total_effect / self.total_count).item()
