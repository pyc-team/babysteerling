from typing import Type, Union

import torch
from torch import nn as nn
from torch.distributions import OneHotCategorical, Bernoulli
from torch.nn import Linear
from torch_concepts import ConceptVariable, Annotations
import torch_concepts as pyc
from torch_concepts.nn import (
    ParametricCPD,
    BayesianNetwork,
    BaseInference,
)

from .alm import ALM
from .lm import ILM
from ..loss import LossOutput


class ConceptNaiveResidualALM(ALM, ILM):
    """Autoregressive language model with concept bottleneck and residual."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int,
        num_heads: int,
        num_kv_heads: int,
        n_layers: int,
        dropout: float,
        loss_fn: nn.Module,
        inference: Type[BaseInference],
        # bottleneck parameters
        out_concepts: Union[int, Annotations],
        tie_weights: bool = True,
        inference_kwargs: dict | None = None,
        steering_every_n_steps: int = 10,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            block_size,
            n_embed,
            num_heads,
            num_kv_heads,
            n_layers,
            dropout,
            loss_fn,
            inference,
            tie_weights,
            inference_kwargs,
            **kwargs,
        )
        if isinstance(out_concepts, int):
            out_concepts = [f"concept_{i}" for i in range(out_concepts)]
            n_concepts = len(out_concepts)
        else:
            n_concepts = len(out_concepts)

        self.steering_every_n_steps = steering_every_n_steps

        self._bottleneck = nn.ModuleDict(
            {
                "concepts": pyc.nn.LinearEmbeddingToConcept(n_embed, n_concepts),
            }
        )
        self._head = Linear(self.n_embed + n_concepts, self.vocab_size)

        self.concepts = ConceptVariable(
            "concepts", distribution=Bernoulli, size=1, members=list(out_concepts)
        )
        self.next_token = ConceptVariable(
            "next_token",
            distribution=OneHotCategorical,
            size=1,
            members=[f"token_{i}" for i in range(vocab_size)],
        )

        self.concepts_cpd = ParametricCPD(
            self.concepts,
            parametrization={"logits": self._bottleneck["concepts"]},
            parents=[self.latent_var],
        )
        self.next_token_cpd = ParametricCPD(
            self.next_token,
            parametrization={"logits": self._head},
            parents=[self.concepts, self.latent_var],
        )

        self.pgm = BayesianNetwork(
            [self.input_var, self.latent_var, self.concepts, self.next_token],
            [self.input_cpd, self.latent_cpd, self.concepts_cpd, self.next_token_cpd],
        )
        self.inference = inference(self.pgm, **(inference_kwargs or {}))

    @property
    def head(self) -> nn.Module:
        return self._head

    @property
    def bottleneck(self) -> nn.ModuleDict:
        return self._bottleneck

    def step(self, batch: dict, *args, **kwargs) -> LossOutput:
        x = self.tokens_to_embedding(batch["input_ids"])

        out_concepts = self.inference.query(
            query=["concepts", "latent"],
            evidence={"input": x},
        )
        total_loss = self.loss_fn(out_concepts, batch, ["concept"], ["concept_auc"])

        concepts = torch.sigmoid(out_concepts.logits["concepts"].tensor)
        if self.training:
            concepts = batch["known_labels"]  # teacher forcing

        out_next_token = self.inference.query(
            query=["next_token"],
            evidence={
                "concepts": concepts,
                "latent": out_concepts.value["latent"].tensor,
            },
        )
        total_loss += self.loss_fn(out_next_token, batch, ["token"], ["token_accuracy"])

        # only to diagnose the model: compute causal effect of intervening on concepts
        B, T, C = batch["known_labels"].shape
        intervention_mask = batch["random_intervention_ids"] == torch.arange(
            C, device=batch["random_intervention_ids"].device
        )
        intervened_labels_1 = torch.where(intervention_mask, 1.0, batch["known_labels"])
        output_1 = self.inference.query(
            query=["next_token"],
            evidence={
                "concepts": intervened_labels_1,
                "latent": out_concepts.value["latent"].tensor,
            },
        )
        intervened_labels_0 = torch.where(intervention_mask, 0.0, batch["known_labels"])
        output_0 = self.inference.query(
            query=["next_token"],
            evidence={
                "concepts": intervened_labels_0,
                "latent": out_concepts.value["latent"].tensor,
            },
        )
        total_loss += self.loss_fn(
            {"next_token_1": output_1, "next_token_0": output_0},
            batch,
            None,
            ["causal_concept_effect"],
        )

        return total_loss
