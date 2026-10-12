import torch
import torch.nn as nn
from torch.nn import functional as F


class Weighted(nn.Module):
    """Binds a fixed weight to a loss at construction"""

    def __init__(self, module: nn.Module, weight: float = 1.0):
        super().__init__()
        self.module = module
        self.weight = weight

    def forward(self, **kwargs) -> torch.Tensor:
        return self.weight * self.module(**kwargs)


class NullLoss(nn.Module):
    """No-op stand-in for an unconfigured loss term"""

    def forward(self, **kwargs) -> None:
        return None


class TokenLoss(nn.Module):
    """Token-level cross-entropy loss."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        mask_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            B, T, C = prediction.shape
            return F.cross_entropy(
                prediction.view(B * T, C),
                target.view(B * T),
                ignore_index=self.ignore_index,
            )

        if target is None or mask.sum() == 0:
            return torch.tensor(0.0, device=prediction.device)

        # Masked cross entropy: [n_masked, vocab] vs [n_masked]
        ce = F.cross_entropy(prediction[mask], target[mask], reduction="none")

        if mask_weights is None:
            return ce.mean()

        # Weighted mean: weight each position by 1 / p_mask
        w = 1.0 / mask_weights[mask]
        return (ce * w).sum() / w.sum()


class ConceptLoss(nn.Module):
    """Document-level soft-OR loss over predicted known concepts."""

    def forward(
        self,
        concept_logits: torch.Tensor,
        doc_spans: list,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if concept_logits is None:
            return torch.tensor(0.0)
        if not doc_spans or (mask is not None and mask.sum() == 0):
            return torch.tensor(0.0, device=concept_logits.device)

        n = concept_logits.shape[-1]
        total = concept_logits.new_zeros(())
        valid_spans = 0

        for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
            if mask is not None:
                # Logits for MASKED tokens only [n_masked, n]; skip spans with none masked
                span_mask = mask[batch_idx, tok_start:tok_end]
                if not span_mask.any():
                    continue
                z_span = concept_logits[batch_idx, tok_start:tok_end, :][span_mask]
            else:
                # Span logits shape: [doc_len, n]
                z_span = concept_logits[batch_idx, tok_start:tok_end, :]

            # log P(concept absent) = sum over tokens of -softplus(z_i)
            # strictly negative for all unconstrained logits z_i
            doc_len = z_span.shape[0]
            # Scaling softplus sum by length (or sqrt(length)) prevents long spans from dominating
            log_p_none = -F.softplus(z_span).sum(dim=0) / (doc_len**0.5)

            # Ensure log_p_none is strictly negative (<= -1e-7) so exp(log_p_none) < 1.0,
            # preventing log1p(-1.0) = log(0)
            log_p_none_safe = log_p_none.clamp(max=-1e-7)

            # log P(concept present somewhere in span) = log(1 - P(none))
            log_p_any = torch.log1p(-torch.exp(log_p_none_safe))  # shape: [n]

            y = concept_logits.new_zeros(n)  # multi-hot ground truth label
            y[concept_ids] = 1.0

            pos_mask = y == 1.0
            neg_mask = y == 0.0

            # Average per positive concept
            pos_loss = (
                -log_p_any[pos_mask].mean()
                if pos_mask.any()
                else concept_logits.new_tensor(0.0)
            )

            # Average per negative concept
            neg_loss = (
                -log_p_none[neg_mask].mean()
                if neg_mask.any()
                else concept_logits.new_tensor(0.0)
            )

            # Each group contributes equally (1:1 weight) regardless of how sparse y is
            total = total + (pos_loss + neg_loss)
            valid_spans += 1

        # Non-diffusion: every span counts (valid_spans == len(doc_spans), none skipped).
        # Diffusion: only spans with masked tokens count; max(,1) guards an all-skipped batch.
        return total / (len(doc_spans) if mask is None else max(valid_spans, 1))


class ReconstructionLoss(nn.Module):
    """MSE between residual representation and target."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction is None or target is None:
            device = prediction.device if prediction is not None else target.device
            return torch.tensor(0.0, device=device)

        if mask is None:
            return ((prediction - target) ** 2).mean()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=prediction.device)

        return ((prediction[mask] - target[mask]) ** 2).mean()


class IndependenceLoss(nn.Module):
    """Penalizes correlation between two representations, so one head doesn't just re-learn
    what the other already captures.
    """

    MAX_VALUE = (
        1.0  # ceiling, so a single large-covariance batch can't dominate the total loss
    )

    def forward(
        self, known_repr: torch.Tensor, other_repr: torch.Tensor
    ) -> torch.Tensor:
        if known_repr is None or other_repr is None:
            device = known_repr.device if known_repr is not None else other_repr.device
            return torch.tensor(0.0, device=device)

        d = known_repr.shape[-1]
        Hk = known_repr.detach().reshape(
            -1, d
        )  # shape: [B, T, d] -> [B*T, d], flatten batch+time into one axis of "samples"
        Hu = other_repr.reshape(-1, d)  # shape: [B, T, d] -> [B*T, d]
        num_tokens = Hk.shape[0]

        Phi = Hk - Hk.mean(
            dim=0, keepdim=True
        )  # shape: [B*T, d], center each feature across the batch
        Psi = Hu - Hu.mean(dim=0, keepdim=True)  # shape: [B*T, d]
        cross_cov = (
            Psi.t() @ Phi
        )  # shape: [d, B*T] @ [B*T, d] -> [d, d], empirical cross-covariance matrix
        hsic = (cross_cov**2).sum() / (
            d**2 * max(num_tokens - 1, 1)
        )  # normalized Frobenius norm^2
        return hsic.clamp(max=self.MAX_VALUE)


# Losses for interventions


class RespondLoss(nn.Module):
    """Eq. 31: Pushes the injected concept's activation alpha_c toward 1 at its attributed positions, training the concept module to recognize
    the concept it was guided toward."""

    def forward(
        self,
        concept_logits: torch.Tensor,
        concept_ids: torch.Tensor | None,
        position_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if concept_logits is None:
            return torch.tensor(0.0)
        if position_mask is None or concept_ids is None or position_mask.sum() == 0:
            return torch.tensor(0.0, device=concept_logits.device)

        B, T, n = concept_logits.shape
        idx = concept_ids.reshape(B, 1, 1).expand(B, T, 1).long()  # shape: [B, T, 1]
        z_c = concept_logits.gather(-1, idx).squeeze(
            -1
        )  # shape: [B, T], each row's own intervened concept logit

        return F.softplus(-z_c[position_mask]).mean()


class ExpressLoss(nn.Module):
    """Eq. 32: Pushes the model's output token distribution toward the concept's lifted tokens at attributed positions."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        next_token_logits: torch.Tensor,
        lifted_token_ids: torch.Tensor | None,
        position_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            position_mask is None
            or lifted_token_ids is None
            or len(lifted_token_ids) == 0
            or position_mask.sum() == 0
        ):
            return torch.tensor(0.0, device=next_token_logits.device)

        probs = F.softmax(next_token_logits, dim=-1)  # shape: [B, T, vocab]
        mass = torch.take_along_dim(probs, lifted_token_ids.long(), dim=-1).sum(
            dim=-1
        )  # shape: [B, T]
        return -torch.log(mass[position_mask].clamp(self.eps, 1.0)).mean()
