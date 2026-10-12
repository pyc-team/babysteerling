from typing import Type

import torch
from torch import nn as nn
from torch.distributions import OneHotCategorical, Bernoulli
from torch.nn import Linear, Sequential
from torch_concepts import ConceptVariable
from torch_concepts.nn import (
    ParametricCPD,
    BayesianNetwork,
    LinearEmbeddingToConcept,
    BaseInference,
)

from ..backbone.lm import ILM
from ..loss import NullLoss
from ..metric import NullMetric
from .utils import _out_concepts_and_n


class ConceptBottleneckMixin(ILM):
    """Vanilla concept bottleneck: next token is predicted from concepts alone."""

    def __init__(
        self,
        out_concepts,
        inference: Type[BaseInference],
        steering_every_n_steps: int = 10,
        inference_kwargs: dict | None = None,
        token_loss: nn.Module | None = None,
        concept_loss: nn.Module | None = None,
        steering_loss: nn.Module | None = None,
        token_metric: nn.Module | None = None,
        concept_metric: nn.Module | None = None,
        causal_concept_effect_metric: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(
            inference=inference, inference_kwargs=inference_kwargs, **kwargs
        )
        out_concepts, n_concepts = _out_concepts_and_n(out_concepts)
        self.steering_every_n_steps = steering_every_n_steps

        self.token_loss = token_loss or NullLoss()
        self.concept_loss = concept_loss or NullLoss()
        self.steering_loss = steering_loss or NullLoss()
        self.token_metric = token_metric or NullMetric()
        self.concept_metric = concept_metric or NullMetric()
        self.causal_concept_effect_metric = causal_concept_effect_metric or NullMetric()

        self._bottleneck = nn.ModuleDict(
            {"known": LinearEmbeddingToConcept(self.n_embed, n_concepts)}
        )
        self._head = Sequential(
            Linear(n_concepts, self.n_embed),
            nn.GELU(),
            Linear(self.n_embed, self.n_embed),
            nn.GELU(),
            Linear(self.n_embed, self.vocab_size),
        )

        self.concepts = ConceptVariable(
            "concepts", distribution=Bernoulli, size=1, members=list(out_concepts)
        )
        self.next_token = ConceptVariable(
            "next_token",
            distribution=OneHotCategorical,
            size=1,
            members=[f"token_{i}" for i in range(self.vocab_size)],
        )

        self.concepts_cpd = ParametricCPD(
            self.concepts,
            parametrization={"logits": self._bottleneck["known"]},
            parents=[self.latent_var],
        )
        self.head_cpd = ParametricCPD(
            self.next_token,
            parametrization={"logits": self._head},
            parents=[self.concepts],
        )

        self.pgm = BayesianNetwork(
            [self.input_var, self.latent_var, self.concepts, self.next_token],
            [self.input_cpd, self.latent_cpd, self.concepts_cpd, self.head_cpd],
        )
        self.inference = inference(self.pgm, **(inference_kwargs or {}))

    @property
    def head(self) -> nn.Module:
        return self._head

    @property
    def bottleneck(self) -> nn.ModuleDict:
        return self._bottleneck

    @property
    def metrics(self) -> dict:
        return {
            "token": self.token_metric,
            "concept": self.concept_metric,
            "causal_concept_effect": self.causal_concept_effect_metric,
        }

    def step(self, batch: dict, *args, **kwargs) -> dict[str, torch.Tensor]:
        x, extras = self.prepare_input(batch)
        mask = extras.get("mask")
        mask_weights = extras.get("p_mask")
        doc_spans = batch.get("doc_spans")
        t_targets = batch.get("targets")
        intervention_ids = batch.get("random_intervention_ids")
        lifted_token_ids = batch.get("lifted_tokens_intervention_ids")
        known_labels = batch.get("known_labels")
        B, T, C = known_labels.shape
        intervention_mask = intervention_ids == torch.arange(
            C, device=intervention_ids.device
        )
        intervened_labels_1 = torch.where(intervention_mask, 1.0, known_labels)
        intervened_labels_0 = torch.where(intervention_mask, 0.0, known_labels)
        losses = {}

        # inference
        output = self.inference.query(query=["concepts"], evidence={"input": x})
        c_logits = output.logits["concepts"].tensor
        c_probs = (
            torch.sigmoid(c_logits) if not self.training else known_labels
        )  # teacher forcing
        output = self.inference.query(
            query=["next_token"], evidence={"concepts": c_probs}
        )
        t_logits = output.logits["next_token"]

        # loss
        losses["concept"] = self.concept_loss(
            concept_logits=c_logits, doc_spans=doc_spans, mask=mask
        )
        losses["token"] = self.token_loss(
            prediction=t_logits, target=t_targets, mask=mask, mask_weights=mask_weights
        )

        # steering
        if self.training and kwargs["step"] % self.steering_every_n_steps == 0:
            output = self.inference.query(
                query=["next_token"], evidence={"concepts": intervened_labels_1}
            )
            losses["steering"] = self.steering_loss(
                next_token_logits=output.logits["next_token"],
                lifted_token_ids=lifted_token_ids,
                position_mask=mask,
            )

        # metrics
        if not self.training:
            # only to diagnose the model: compute causal effect of intervening on concepts
            output_1 = self.inference.query(
                query=["next_token"], evidence={"concepts": intervened_labels_1}
            )
            output_0 = self.inference.query(
                query=["next_token"], evidence={"concepts": intervened_labels_0}
            )
            logits_1, logits_0 = (
                output_1.logits["next_token"],
                output_0.logits["next_token"],
            )

            self.causal_concept_effect_metric.update(
                logits_1=logits_1, logits_0=logits_0
            )
            self.concept_metric.update(concept_logits=c_logits, doc_spans=doc_spans)
            self.token_metric.update(prediction=t_logits, target=t_targets)

        return {k: v for k, v in losses.items() if v is not None}
