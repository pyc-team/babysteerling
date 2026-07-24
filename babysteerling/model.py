"""Model architecture: causal transformer backbone + Steerling concept bottleneck + head.

Independently implements the architecture described in Guide Labs' technical report "Scaling
Inherently Interpretable Language Models" (see the project's NOTICE for attribution) at a scale
that trains in minutes on a laptop. The concept bottleneck decomposes the backbone's hidden state
into three additive, inspectable pieces before the final projection to vocabulary logits:

    h_bar = k_hat (known concepts) + u_hat (unknown concepts) + epsilon (residual)

Intuition: instead of letting the model use its hidden state however it wants, we force part of
it to route through a small set of human-labeled "known" concepts (supervised by a concept
library -- see babysteerling.data.atlas for how to build one), part through a larger set of free
"unknown" concepts the model discovers on its own, and let a residual mop up whatever's left.
Because the final head is linear, every output logit is then an exact sum of a known-concept
contribution, an unknown-concept contribution, and a residual contribution -- which is what makes
the model's predictions attributable back to specific concepts.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .nn.backbone import TransformerModel
from .nn.bottleneck import ConceptBottleneck
from .nn.predictor import LinearEmbeddingToConcept, ReluEmbeddingToConcepts


class SteerlingGPT(nn.Module):
    """Backbone + concept bottleneck + head, as one module.

    Wrapping everything in a single nn.Module (rather than passing three separate objects
    around) means model.parameters() and model.state_dict() naturally deduplicate the tied
    embedding/head weight via PyTorch's built-in traversal.

    backbone_type="causal" (default): plain next-token-prediction attention (see
    MultiHeadAttention). backbone_type="diffusion": block-causal attention (bidirectional within
    a block of `diff_block_len` tokens, causal across blocks) for use with the masked-diffusion
    training objective in babysteerling.diffusion. This class only needs to know which mask (if
    any) attention should use -- it builds that mask once at construction time and stores it as
    a buffer; it doesn't otherwise know or care about corruption/sampling, which live entirely
    in babysteerling.diffusion.
    """

    def __init__(self, vocab_size, block_size, n_embed, num_heads, num_kv_heads, n_layers,
                 dropout, n_concepts, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, head_type="linear", tie_weights=True,
                 head_mlp_hidden=None, backbone_type="causal"):
        super().__init__()
        self.block_size = block_size
        self.backbone_type = backbone_type
        self.backbone = TransformerModel(vocab_size, n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads)
        self.bottleneck = ConceptBottleneck(
            n_embed, n_concepts, unknown_ratio=unknown_ratio, p_epsilon=p_epsilon,
            unknown_rank=unknown_rank, top_k_known=top_k_known, top_k_unknown=top_k_unknown,
        )
        nn.init.constant_(self.bottleneck.known.g[-1].bias, -math.log(block_size))
        tied_embedding = self.backbone.token_embedding_table.weight if tie_weights else None
        if head_type == "linear":
            self.head = LinearEmbeddingToConcept(
                n_embed, vocab_size, tie_weights=tie_weights, tied_embedding=tied_embedding,
            )
        elif head_type == "mlp":
            self.head = ReluEmbeddingToConcepts(
                n_embed, vocab_size, mlp_hidden=head_mlp_hidden,
            )

        attn_mask = None
        # non-persistent: it's cheap to rebuild and shouldn't be saved into/loaded from checkpoints
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, idx, known_labels=None):
        h = self.backbone(idx, attn_mask=self.attn_mask)  # shape: [B, T, n_embed]
        h_bar, intermediates = self.bottleneck(h, known_labels=known_labels)
        logits = self.head(h_bar)  # shape: [B, T, vocab_size]
        return logits, intermediates

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling, one token at a time, cropping context to block_size."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # shape: [B, <=block_size], keep only the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # shape: [B, T, vocab] -> [B, vocab], last-position logits only

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # shape: [B, top_k], top-k logit values
                logits[logits < v[:, [-1]]] = float('-inf')  # mask out everything below the k-th largest logit

            probs = F.softmax(logits, dim=-1)  # shape: [B, vocab]
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: [B, 1]
            idx = torch.cat((idx, idx_next), dim=1)  # shape: [B, T] -> [B, T+1]
        return idx


class SteerlingDiffusion(SteerlingGPT):
    """SteerlingGPT with diffusion-specific forward() signature.

    The only differences are the forward and the generate methods: the diffusion forward() takes a `corruption_mask`
    argument and returns the corrupted input and the corruption mask, while the generate() method is adapted
    for diffusion sampling.
    """

    def __init__(self, vocab_size, block_size, n_embed, num_heads, num_kv_heads, n_layers,
                 dropout, n_concepts, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, head_type="linear", tie_weights=True,
                 head_mlp_hidden=None, backbone_type="causal", diff_block_len=None):
        super().__init__(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embed=n_embed,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_concepts=n_concepts,
            unknown_ratio=unknown_ratio,
            p_epsilon=p_epsilon,
            unknown_rank=unknown_rank,
            top_k_known=top_k_known,
            top_k_unknown=top_k_unknown,
            head_type=head_type,
            tie_weights=tie_weights,
            head_mlp_hidden=head_mlp_hidden,
            backbone_type=backbone_type
        )
        assert diff_block_len is not None, "diff_block_len is required when backbone_type='diffusion'"
        from .diffusion import build_block_causal_mask  # local import: diffusion.py doesn't need to import nn.py
        attn_mask = build_block_causal_mask(block_size, diff_block_len)

        # non-persistent: it's cheap to rebuild and shouldn't be saved into/loaded from checkpoints
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    @torch.no_grad()
    def generate(model, mask_token_id, seq_len, vocab_size, gen_steps=32, temperature=0.8, top_k=50):
        """Basic random-remasking MDM sampler for sanity-checking output only -- not the paper's
        efficient block-wise KV-cached inference procedure (out of scope here). Operates on any
        model exposing SteerlingGPT's forward(idx) -> (logits, intermediates) interface; doesn't
        need to know anything about the concept bottleneck.
        """
        device = next(model.parameters()).device
        model.eval()
        x = torch.full((1, seq_len), mask_token_id, dtype=torch.long,
                       device=device)  # shape: [1, seq_len], start fully masked
        masked = torch.ones_like(x, dtype=torch.bool)  # shape: [1, seq_len], tracks which positions are still masked

        for step in range(1, gen_steps + 1):
            logits, _ = model(x)  # shape: [1, seq_len, vocab_size]
            target_masked_count = round(
                seq_len * (1 - step / gen_steps))  # shrink the masked budget linearly over gen_steps

            probs = F.softmax(logits / temperature, dim=-1)  # shape: [1, seq_len, vocab_size]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)  # shape: [1, seq_len, top_k]
                probs = torch.where(logits < v[..., [-1]], torch.zeros_like(probs), probs)  # zero out everything below the k-th largest logit
                probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize after truncation
            sampled = torch.multinomial(probs.view(-1, vocab_size), 1).view(1,
                                                                            seq_len)  # shape: [seq_len, vocab_size] -> [seq_len, 1] -> [1, seq_len]

            masked_positions = masked[0].nonzero(as_tuple=True)[
                0]  # shape: [n_still_masked], indices of masked positions
            num_to_reveal = max(len(masked_positions) - target_masked_count, 0)
            if num_to_reveal > 0:
                # reveal a random subset of currently-masked positions (not necessarily the most
                # confident ones) -- simplest possible sampler, good enough for a sanity check
                reveal_idx = masked_positions[torch.randperm(len(masked_positions))[:num_to_reveal]]
                x[0, reveal_idx] = sampled[0, reveal_idx]
                masked[0, reveal_idx] = False

        model.train()
        return x[0].tolist()


def build_model(vocab_size, n_concepts, block_size, n_embed=128, num_heads=4, num_kv_heads=2,
                 n_layers=4, dropout=0.2, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, head_type="linear", tie_weights=True,
                 head_mlp_hidden=None, backbone_type="causal", diff_block_len=None):
    """Factory: construct a SteerlingGPT from plain keyword arguments (defaults match this
    project's baseline config). Takes no framework-specific config object, so it can be called
    the same way whether or not the caller uses Hydra -- see experiments/train.py for the
    Hydra-config adapter that unpacks a `cfg.model` group into this call.
    """
    if backbone_type.lower() == "causal":
        return SteerlingGPT(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embed=n_embed,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_concepts=n_concepts,
            unknown_ratio=unknown_ratio,
            p_epsilon=p_epsilon,
            unknown_rank=unknown_rank,
            top_k_known=top_k_known,
            top_k_unknown=top_k_unknown,
            head_type=head_type,
            tie_weights=tie_weights,
            head_mlp_hidden=head_mlp_hidden,
            backbone_type=backbone_type,
        )
    elif backbone_type.lower() == "diffusion":
        return SteerlingDiffusion(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embed=n_embed,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_concepts=n_concepts,
            unknown_ratio=unknown_ratio,
            p_epsilon=p_epsilon,
            unknown_rank=unknown_rank,
            top_k_known=top_k_known,
            top_k_unknown=top_k_unknown,
            head_type=head_type,
            tie_weights=tie_weights,
            head_mlp_hidden=head_mlp_hidden,
            backbone_type=backbone_type,
            diff_block_len=diff_block_len
        )
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}. Supported types are 'causal' and 'diffusion'.")
