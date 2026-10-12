from typing import Type

import torch
from torch import nn
from torch.distributions import OneHotCategorical
from torch.nn import functional as F
from torch_concepts import ConceptVariable
from torch_concepts.nn import ParametricCPD, BayesianNetwork, BaseInference

from .lm import LM
from ..loss import NullLoss
from ..metric import NullMetric
from ..nn.predictor import LinearEmbeddingToConcept


class ALM(LM):
    """Autoregressive language model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int,
        num_heads: int,
        num_kv_heads: int,
        n_layers: int,
        dropout: float,
        inference: Type[BaseInference],
        tie_weights: bool = True,
        inference_kwargs: dict | None = None,
        token_loss: nn.Module | None = None,
        token_metric: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embed=n_embed,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs,
        )
        self.token_loss = token_loss or NullLoss()
        self.token_metric = token_metric or NullMetric()
        self.tie_weights = tie_weights
        tied_embedding = (
            self.tokens_to_embedding.token_embedding_table.weight
            if tie_weights
            else None
        )
        self._head = LinearEmbeddingToConcept(
            self.n_embed,
            self.vocab_size,
            tie_weights=tie_weights,
            tied_embedding=tied_embedding,
        )

        self.next_token = ConceptVariable(
            "next_token",
            distribution=OneHotCategorical,
            size=1,
            members=[f"token_{i}" for i in range(vocab_size)],
        )
        self.head_cpd = ParametricCPD(
            self.next_token,
            parametrization={"logits": self._head},
            parents=[self.latent_var],
        )

        self.pgm = BayesianNetwork(
            [self.input_var, self.latent_var, self.next_token],
            [self.input_cpd, self.latent_cpd, self.head_cpd],
        )
        self.inference = inference(self.pgm, **(inference_kwargs or {}))

    @property
    def head(self) -> nn.Module:
        return self._head

    def prepare_input(self, batch: dict) -> tuple[torch.Tensor, dict]:
        return self.tokens_to_embedding(batch["input_ids"]), {}

    @property
    def metrics(self) -> dict:
        return {"token": self.token_metric}

    def step(self, batch: dict, **kwargs) -> dict[str, torch.Tensor]:
        x, extras = self.prepare_input(batch)

        output = self.inference.query(
            query=["next_token"],
            evidence={"input": x},
        )

        if not self.training:
            self.token_metric.update(
                prediction=output.logits["next_token"], target=batch["targets"]
            )

        losses = {
            "token": self.token_loss(
                prediction=output.logits["next_token"],
                target=batch["targets"],
                mask=extras.get("mask"),
                mask_weights=extras.get("p_mask"),
            )
        }
        return {k: v for k, v in losses.items() if v is not None}

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        prompt: torch.Tensor | None = None,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 50,
        **kwargs,
    ):
        """Autoregressive sampling, one token at a time, cropping context to block_size."""
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        idx = (
            torch.zeros((batch_size, 2), dtype=torch.long, device=device)
            if prompt is None
            else prompt.to(device)
        )

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            x = self.tokens_to_embedding(idx_cond)
            out = self.inference.query(query=["next_token"], evidence={"input": x})
            logits = out.logits["next_token"][:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()

        return idx[0].tolist()
