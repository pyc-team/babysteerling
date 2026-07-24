"""Loss functions for the concept-bottleneck model.

Deliberately pure: every function/class here only consumes logits/intermediates already
produced by nn.py's forward pass -- nothing in this file runs the model. That split keeps
"what the model computes" and "how we score what it computed" independently editable.

Four terms are combined into the total training loss:
  1. Language modeling loss  -- the actual next-token prediction task (kept inline in train.py,
     since it's just F.cross_entropy on the head's output; no need for a wrapper class).
  2. Concept loss            -- did the known-concept head predict the right concepts?
  3. Reconstruction loss     -- does the unknown head's embedding match what's left of h once
     the known concepts are subtracted out (i.e. is it capturing genuinely complementary info)?
  4. Independence loss       -- are the known and unknown representations decorrelated, so they
     don't both encode the same information redundantly?
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class ConceptLoss(nn.Module):
    """OR-aggregated binary cross-entropy over per-document concept labels.

    Intuition: a document's ground-truth label says "concept c appears somewhere in this
    document", not "concept c appears at every token". So we aggregate the per-token
    predicted activation into a single per-document probability via a soft-OR (1 - product of
    "concept absent" probabilities across tokens) before comparing to the binary label -- the
    loss is satisfied as soon as the concept is confidently predicted at *any* token in the span.
    Computing the same expression from logits avoids multiplying saturated probabilities.
    """

    def forward(self, k_logits, doc_spans):
        """
        k_logits: [B, T, n] predicted known-concept logits.
        doc_spans: list of (batch_idx, tok_start, tok_end, concept_ids), one per document
            overlapping the current batch of windows (see data/utils.py's build_supervision()).
        """
        if not doc_spans:
            return k_logits.new_zeros(())
        n = k_logits.shape[-1]
        total = k_logits.new_zeros(())
        for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
            # softplus(logit) is the negative log-probability that a concept is absent
            evidence = F.softplus(k_logits[batch_idx, tok_start:tok_end]).sum(dim=0)
            y = torch.zeros(n, dtype=torch.bool, device=k_logits.device)
            y[concept_ids] = True
            total = total + (
                -torch.log(-torch.expm1(-evidence[y])).sum()
                + evidence[~y].sum()
            ) / n
        return total / len(doc_spans)  # average per-document loss, so batch size doesn't change the scale


class ReconstructionLoss(nn.Module):
    """MSE between the unknown head's u_hat and its ground-truth target (h minus the known
    concepts' contribution). Trains the unknown head to capture exactly what the known
    concepts don't, rather than an arbitrary or redundant transformation of h.

    `mask`, if given, restricts the loss to those positions only (e.g. the diffusion backbone's
    corruption mask -- only masked positions have a meaningful prediction target there). `None`
    (default) uses every position, correct for the causal backbone where every position does.
    """

    def forward(self, u_hat, u_hat_gt, mask=None):
        if u_hat_gt is None:
            return torch.tensor(0.0, device=u_hat.device)
        if mask is None:
            return ((u_hat - u_hat_gt) ** 2).mean()  # shape: [B, T, d] -> scalar
        if mask.sum() == 0:
            return torch.tensor(0.0, device=u_hat.device)
        diff = u_hat[mask] - u_hat_gt[mask]  # shape: [B, T, d] -> [n_masked, d]
        return (diff ** 2).mean()


class IndependenceLoss(nn.Module):
    """Cross-covariance penalty between k_hat and u_hat (linear-kernel HSIC-style term).

    Intuition: without this, the unknown head is free to re-derive the same information the
    known head already captures, wasting capacity on redundancy instead of complementary
    information. Penalizing the (linear) statistical dependence between the two pushes them
    toward encoding different things. Gradients only flow through the unknown side (k_hat is
    detached) -- we want the *unknown* head to adapt to the known head, not the other way
    around, since the known head is anchored to human-labeled concepts.
    """

    def forward(self, k_hat, u_hat):
        d = k_hat.shape[-1]
        Hk = k_hat.detach().reshape(-1, d)  # shape: [B, T, d] -> [B*T, d], flatten batch+time into one axis of "samples"
        Hu = u_hat.reshape(-1, d)  # shape: [B, T, d] -> [B*T, d]
        num_tokens = Hk.shape[0]

        Phi = Hk - Hk.mean(dim=0, keepdim=True)  # shape: [B*T, d], center each feature across the batch
        Psi = Hu - Hu.mean(dim=0, keepdim=True)  # shape: [B*T, d]
        cross_cov = Psi.t() @ Phi  # shape: [d, B*T] @ [B*T, d] -> [d, d], empirical cross-covariance matrix
        return (cross_cov ** 2).sum() / (d ** 2 * max(num_tokens - 1, 1))  # normalized Frobenius norm^2


# module-level singletons: these losses hold no learnable parameters, so one shared instance
# (rather than constructing a fresh one on every call) is enough
_concept_loss_fn = ConceptLoss()
_rec_loss_fn = ReconstructionLoss()
_indep_loss_fn = IndependenceLoss()


def compute_losses(logits, targets, intermediates, doc_spans,
                    lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0, mask=None):
    """Combine all four loss terms into the total training objective.

    logits/targets: the model's token predictions and the ground-truth tokens they should
    match. `mask`, if given (e.g. the diffusion backbone's corruption mask from
    babysteerling.diffusion.corrupt), restricts the LM and reconstruction losses to those
    positions only -- the ones that actually had something to predict. `None` (default) scores
    every position, correct for the causal backbone's next-token objective.
    intermediates: the dict returned by nn.py's ConceptBottleneck.forward().
    Returns (total_loss, components) where components is a plain dict of floats, handy for
    logging each term separately (console / W&B) without re-running the forward pass.
    """
    B, T, C = logits.shape
    if mask is None:
        lm_loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))  # shape: [B, T, vocab] -> [B*T, vocab] vs [B*T]
    elif mask.sum() == 0:
        lm_loss = torch.tensor(0.0, device=logits.device)
    else:
        lm_loss = F.cross_entropy(logits[mask], targets[mask])  # shape: [B, T, vocab] -> [n_masked, vocab] vs [n_masked]

    concept_loss = _concept_loss_fn(intermediates['k_logits'], doc_spans)
    rec_loss = _rec_loss_fn(intermediates['u_hat'], intermediates['u_hat_gt'], mask=mask)
    indep_loss = _indep_loss_fn(intermediates['k_hat'], intermediates['u_hat'])

    total_loss = lm_loss + lambda_concept * concept_loss + lambda_rec * rec_loss + lambda_indep * indep_loss
    components = {
        'total': total_loss.item(),
        'lm': lm_loss.item(),
        'concept': concept_loss.item(),
        'rec': rec_loss.item(),
        'indep': indep_loss.item(),
    }
    return total_loss, components
