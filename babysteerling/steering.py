import random
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_concepts.nn import ConceptInterventionStrategy, UniformPolicy


def _flatten_to_2d(x):
    """[B, T, F] -> [B*T, F] (a 2-D [B, F] input is left as is). Returns (flat, original_shape)
    so the caller can restore it after. pytorch_concepts' build_mask() picks rows by quantile,
    so treating each token as its own row here is correct, not a hack.
    """
    if x.dim() == 2:
        return x, None
    if x.dim() == 3:
        B, T, F_ = x.shape
        return x.reshape(B * T, F_), (B, T, F_)
    raise ValueError(
        f"expected a 2-D [B,F] or 3-D [B,T,F] tensor, got shape {tuple(x.shape)}"
    )


def _unflatten(x, original_shape):
    if original_shape is None:
        return x
    B, T, F_ = original_shape
    return x.reshape(B, T, F_)


class AddDirectionStrategy(ConceptInterventionStrategy):
    """x -> x + gamma*direction (Eq. 18). pytorch_concepts' own strategies replace a value
    outright; this one adds a direction on top of whatever value is already there.
    """

    def __init__(self, direction, gamma):
        super().__init__()
        self.register_buffer(
            "direction", direction.detach()
        )  # buffer, so it moves with .to(device)
        self.gamma = gamma

    def forward(self, x, *args, **kwargs):
        return x + self.gamma * self.direction


class InterventionModule(nn.Module):
    """Wraps a module that outputs [B, T, F] or [B, F], and applies intervention_strategy to
    its output, gated per row by intervention_policy. Not a subclass of pytorch_concepts'
    BaseInterventionModule, since that requires a 2-D [B, F] output and can't wrap a transformer
    block or a per-token activation(). This flattens to 2-D, calls their unmodified
    build_mask(), and flattens back. Works with any of their (or our) Strategy/Policy classes.
    """

    def __init__(
        self,
        original_module,
        intervention_strategy,
        intervention_policy,
        sel_idx=None,
        quantile=1.0,
        eps=1e-12,
    ):
        super().__init__()
        self.original_module = original_module
        self.intervention_strategy = intervention_strategy
        self.intervention_policy = intervention_policy
        self.sel_idx = sel_idx
        self.quantile = quantile
        self.eps = eps

    def forward(self, *args, **kwargs):
        output = self.original_module(*args, **kwargs)
        flat, shape = _flatten_to_2d(output)

        policy_scores = self.intervention_policy(flat)
        # build_mask() uses torch.kthvalue, which MPS doesn't support, so run it on CPU and
        # move the result back. Cheap either way, since policy_scores is only [rows, F].
        mask = self.intervention_policy.build_mask(
            policy_scores.cpu(),
            sel_idx=self.sel_idx,
            quantile=self.quantile,
            eps=self.eps,
        ).to(
            device=flat.device, dtype=flat.dtype
        )  # 1 = keep original, 0 = replace (pytorch_concepts' convention)
        intervened = self.intervention_strategy(flat)

        result = flat * mask + intervened * (1.0 - mask)
        return _unflatten(result, shape)


@contextmanager
def steered(model, direction, gamma, inj_layer):
    """Temporarily wraps model.backbone.blocks[inj_layer:] so every position gets the
    AddDirectionStrategy injection (Eq. 18). Restores the original blocks on exit either way,
    so it's safe to use inside eval/metric code.
    """
    blocks = model.backbone.blocks
    layer_ids = range(inj_layer, len(blocks))
    originals = {i: blocks[i] for i in layer_ids}
    try:
        for i in layer_ids:
            blocks[i] = InterventionModule(
                originals[i],
                AddDirectionStrategy(direction, gamma),
                UniformPolicy(),
                quantile=1.0,
            )
        yield model
    finally:
        for i, orig in originals.items():
            blocks[i] = orig


@contextmanager
def injected_at(model, direction, gamma, position_mask, inj_layer):
    """Like steered(), but injects only at a given position_mask [B, T] instead of everywhere.
    Used by run_steering_batch, where the positions come from positions_for_concept and are
    already known, so no Policy is needed to choose them. Uses a plain forward hook instead of
    InterventionModule, since there's no strategy/policy choice to make here.
    """

    def hook(module, inputs, output):
        # These blocks are shared with prototype-based known encoders (nn.prototype.
        # PrototypePredictor), which re-enter them mid-forward to encode prototype texts at a
        # different length. Skip those calls: only the real batch matches position_mask's shape.
        if output.shape[:2] != position_mask.shape:
            return output
        return inject_at_positions(output, direction, gamma, position_mask)

    handles = [
        block.register_forward_hook(hook)
        for block in list(model.backbone.blocks)[inj_layer:]
    ]
    try:
        yield model
    finally:
        for h in handles:
            h.remove()


def concept_direction(embedding_table, concept_ids):
    """e_c = K_c / ||K_c|| (Eq. 18). Pass a list of concept_ids to steer toward several concepts
    at once (their embeddings are summed then normalized). embedding_table: a head's own
    [n_or_m, D] embedding, e.g. model.bottleneck.known.K.
    """
    if isinstance(concept_ids, int):
        concept_ids = [concept_ids]
    vec = embedding_table[concept_ids].sum(dim=0)  # shape: [len(concept_ids), D] -> [D]
    return vec / vec.norm()


def calibrate_gamma(direction, head, tau=4.0):
    """gamma = tau / peak(e_c), where peak(e_c) = max_y (e_c . W_y) (Eq. 19). Scales the
    injection so its largest effect on any output logit equals tau, giving comparable steering
    strength across concepts without tuning gamma by hand. Needs head_type="linear", since the
    calibration assumes a real linear projection.

    direction may be a single [D] vector shared by the whole batch (returns a float), or a
    per-row [B, D] batch of directions -- e.g. one concept per window (returns a [B] tensor).
    """
    if head.__class__.__name__ == "Linear":
        weight = head.weight  # shape: [vocab, D]
    elif head.head_type == "linear":
        weight = head.head.weight
    else:
        raise ValueError(
            "calibrate_gamma requires head_type='linear' (Eq. 19 assumes a linear LM head)"
        )

    if direction.dim() == 1:
        alignment = (
            weight @ direction
        )  # shape: [vocab, D] @ [D] -> [vocab], e_c . W_y for every y
        peak = alignment.max().item()
        return tau / peak
    alignment = direction @ weight.T  # shape: [B, D] @ [D, vocab] -> [B, vocab]
    peak = alignment.max(dim=-1).values  # shape: [B]
    return tau / peak.clamp(min=1e-6)


def suppress_logits(logits, direction, head, strength):
    """Suppresses a concept in the output: l_v -= strength*ReLU(a_c[v]) for every logit v,
    where a_c = W.e_c is the concept's alignment with each token (Eq. 20-21). The ReLU stops
    this from boosting anti-aligned tokens instead (see paper Figure 20). Acts directly on
    logits, so it's a plain function rather than going through InterventionModule.
    """
    if head.head_type != "linear":
        raise ValueError("suppress_logits requires head_type='linear'")
    a_c = head.head.weight @ direction  # shape: [vocab, D] @ [D] -> [vocab]
    return logits - strength * F.relu(a_c)


def positions_for_concept(token_window, doc_spans, concept_id, lifted_token_ids):
    """Boolean mask [B, T], True where a token (a) is inside a document tagged with
    concept_id (doc_spans, from data.utils.build_supervision), and (b) is one of concept_id's
    lifted tokens (Section 4.4's lift metric). This is the token-level attribution
    steering-training needs, built entirely from data the pipeline already produces.
    """
    doc_mask = torch.zeros_like(token_window, dtype=torch.bool)
    for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
        if concept_id in concept_ids:
            doc_mask[batch_idx, tok_start:tok_end] = True
    if not lifted_token_ids:
        return torch.zeros_like(doc_mask)
    lifted = torch.as_tensor(list(lifted_token_ids), device=token_window.device)
    return doc_mask & torch.isin(token_window, lifted)


def inject_at_positions(h, direction, gamma, position_mask):
    """h + gamma*direction (Eq. 18), only where position_mask is True. Same idea as
    AddDirectionStrategy, but for when the positions are already known instead of chosen by a
    Policy (see injected_at()).

    direction/gamma may be a single [D]/float shared across the batch, or a per-row [B, D]/[B]
    pair -- e.g. one concept per window.
    """
    if direction.dim() == 2:  # [B, D]: one direction (and gamma) per batch row
        delta = (gamma.unsqueeze(-1) * direction).unsqueeze(1)  # [B, D] -> [B, 1, D]
    else:
        delta = gamma * direction  # shape: [D], broadcasts against h's last dim
    return h + position_mask.unsqueeze(-1).to(h.dtype) * delta


def sample_steering_target(doc_spans, lifted_tokens):
    """Picks which concept to steer toward this step: a random choice among concepts present
    in this batch that also have at least one lifted token. This decides which concept to
    target, a different question than a Policy answers (it picks positions within an
    already-chosen target).
    """
    candidates = sorted(
        {
            c
            for _, _, concept_ids in doc_spans
            for c in concept_ids
            if lifted_tokens.get(c)
        }
    )
    if not candidates:
        return None
    return random.choice(candidates)
