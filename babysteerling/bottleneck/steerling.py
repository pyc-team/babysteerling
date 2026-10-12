from functools import reduce
from operator import sub
from typing import Type

import torch
from torch import nn as nn
from torch.distributions import OneHotCategorical, Bernoulli
from torch.nn import functional as F
from torch_concepts import ConceptVariable, EmbeddingVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD,
    BayesianNetwork,
    BaseInference,
)

from ..backbone.lm import ILM
from ..loss import NullLoss
from ..metric import NullMetric
from .utils import _out_concepts_and_n
from .. import steering
from ..nn.encoder import (
    ConceptToLowRankEmbeddings,
    ResidualModule,
    SparseEmbeddingToConcept,
    CallableModule,
)


class SteerlingMixin(ILM):
    """Known + unknown + residual bottleneck"""

    def __init__(
        self,
        out_concepts,
        inference: Type[BaseInference],
        unknown_ratio: float = 3,
        p_epsilon: float = 0.1,
        unknown_rank: int = 32,
        top_k_known: int = None,
        top_k_unknown: int = None,
        inference_kwargs: dict | None = None,
        steering_every_n_steps: int = 10,
        inj_layer: int = 1,
        tau: float = 4.0,
        token_loss: nn.Module | None = None,
        concept_loss: nn.Module | None = None,
        reconstruction_loss: nn.Module | None = None,
        independence_unknown_loss: nn.Module | None = None,
        independence_residual_loss: nn.Module | None = None,
        steering_loss: nn.Module | None = None,
        respond_loss: nn.Module | None = None,
        token_metric: nn.Module | None = None,
        concept_metric: nn.Module | None = None,
        causal_concept_effect_metric: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(
            inference=inference, inference_kwargs=inference_kwargs, **kwargs
        )
        out_concepts, n_concepts = _out_concepts_and_n(out_concepts)

        self.n_unknown_concepts = int(unknown_ratio * n_concepts)
        out_unknown_concepts = [f"unknown_{i}" for i in range(self.n_unknown_concepts)]

        self.steering_every_n_steps = steering_every_n_steps
        self.inj_layer = inj_layer
        self.tau = tau

        self.token_loss = token_loss or NullLoss()
        self.concept_loss = concept_loss or NullLoss()
        self.reconstruction_loss = reconstruction_loss or NullLoss()
        self.independence_unknown_loss = independence_unknown_loss or NullLoss()
        self.independence_residual_loss = independence_residual_loss or NullLoss()
        self.steering_loss = steering_loss or NullLoss()
        self.respond_loss = respond_loss or NullLoss()
        self.token_metric = token_metric or NullMetric()
        self.concept_metric = concept_metric or NullMetric()
        self.causal_concept_effect_metric = causal_concept_effect_metric or NullMetric()

        self._bottleneck = nn.ModuleDict(
            {
                "known": SparseEmbeddingToConcept(
                    self.n_embed, n_concepts, top_k=top_k_known
                ),
                "unknown": SparseEmbeddingToConcept(
                    self.n_embed, self.n_unknown_concepts, top_k=top_k_unknown
                ),
                "known_embeddings": ConceptToLowRankEmbeddings(
                    n_concepts, self.n_embed
                ),
                "unknown_embeddings": ConceptToLowRankEmbeddings(
                    self.n_unknown_concepts, self.n_embed, rank=unknown_rank
                ),
                "unknown_target_embeddings": CallableModule(
                    lambda c: torch.sub(*c.chunk(2, dim=-1))
                ),
                "residual_embeddings": CallableModule(
                    lambda c: reduce(sub, c.chunk(3, dim=-1))
                ),
                "residual": ResidualModule(p_epsilon),
                "reconstruction": CallableModule(lambda c: sum(c.chunk(3, dim=-1))),
            }
        )

        # Define variables
        self.known_concepts = ConceptVariable(
            "concepts", distribution=Bernoulli, size=1, members=list(out_concepts)
        )
        self.unknown_concepts = ConceptVariable(
            "unknown",
            distribution=Bernoulli,
            size=1,
            members=list(out_unknown_concepts),
        )
        self.known_embeddings = EmbeddingVariable(
            "known_embedding", distribution=Delta, size=self.n_embed
        )
        self.unknown_embeddings = EmbeddingVariable(
            "unknown_embedding", distribution=Delta, size=self.n_embed
        )
        self.unknown_target_embeddings = EmbeddingVariable(
            "unknown_target_embedding", distribution=Delta, size=self.n_embed
        )
        self.residual_embeddings = EmbeddingVariable(
            "residual_embedding", distribution=Delta, size=self.n_embed
        )
        self.residual = EmbeddingVariable(
            "residual", distribution=Delta, size=self.n_embed
        )
        self.reconstruction = EmbeddingVariable(
            "reconstruction", distribution=Delta, size=self.n_embed
        )
        self.next_token = ConceptVariable(
            "next_token",
            distribution=OneHotCategorical,
            size=1,
            members=[f"token_{i}" for i in range(self.vocab_size)],
        )

        # Define CPDs
        self.concepts_cpd = ParametricCPD(
            self.known_concepts,
            parametrization={"logits": self._bottleneck["known"]},
            parents=[self.latent_var],
        )
        self.unknown_cpd = ParametricCPD(
            self.unknown_concepts,
            parametrization={"logits": self._bottleneck["unknown"]},
            parents=[self.latent_var],
        )
        self.known_embeddings_cpd = ParametricCPD(
            self.known_embeddings,
            parametrization=self._bottleneck["known_embeddings"],
            parents=[self.known_concepts],
        )
        self.unknown_embeddings_cpd = ParametricCPD(
            self.unknown_embeddings,
            parametrization=self._bottleneck["unknown_embeddings"],
            parents=[self.unknown_concepts],
        )
        self.unknown_target_embeddings_cpd = ParametricCPD(
            self.unknown_target_embeddings,
            parametrization=self._bottleneck["unknown_target_embeddings"],
            parents=[self.latent_var, self.known_embeddings],
        )
        self.residual_embeddings_cpd = ParametricCPD(
            self.residual_embeddings,
            parametrization=self._bottleneck["residual_embeddings"],
            parents=[self.latent_var, self.known_embeddings, self.unknown_embeddings],
        )
        self.residual_cpd = ParametricCPD(
            self.residual,
            parametrization=self._bottleneck["residual"],
            parents=[self.residual_embeddings],
        )
        self.reconstruction_cpd = ParametricCPD(
            self.reconstruction,
            parametrization=self._bottleneck["reconstruction"],
            parents=[self.known_embeddings, self.unknown_embeddings, self.residual],
        )
        self.head_cpd = ParametricCPD(
            self.next_token,
            parametrization={"logits": self._head},
            parents=[self.reconstruction],
        )

        self.pgm = BayesianNetwork(
            [
                self.input_var,
                self.latent_var,
                self.known_concepts,
                self.unknown_concepts,
                self.known_embeddings,
                self.unknown_embeddings,
                self.unknown_target_embeddings,
                self.residual_embeddings,
                self.residual,
                self.reconstruction,
                self.next_token,
            ],
            [
                self.input_cpd,
                self.latent_cpd,
                self.concepts_cpd,
                self.unknown_cpd,
                self.known_embeddings_cpd,
                self.unknown_embeddings_cpd,
                self.unknown_target_embeddings_cpd,
                self.residual_embeddings_cpd,
                self.residual_cpd,
                self.reconstruction_cpd,
                self.head_cpd,
            ],
        )
        self.inference = inference(self.pgm, **(inference_kwargs or {}))

    @property
    def bottleneck(self) -> nn.Module:
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
        output = self.inference.query(
            query=["concepts", "unknown", "latent"], evidence={"input": x}
        )
        c_logits, latent = (
            output.logits["concepts"].tensor,
            output.value["latent"].tensor,
        )
        u_probs = torch.sigmoid(output.logits["unknown"].tensor)
        c_probs = (
            torch.sigmoid(c_logits) if not self.training else known_labels
        )  # teacher forcing
        output = self.inference.query(
            query=["unknown_target_embedding", "known_embedding", "unknown_embedding"],
            evidence={"concepts": c_probs, "unknown": u_probs, "latent": latent},
        )
        unknown_target_embedding = output.value["unknown_target_embedding"].tensor
        known_embedding = output.value["known_embedding"].tensor
        unknown_embedding = output.value["unknown_embedding"].tensor
        output = self.inference.query(
            query=["next_token", "residual_embedding"],
            evidence={
                "known_embedding": known_embedding,
                "unknown_embedding": unknown_embedding,
                "latent": latent,
            },
        )
        t_logits = output.logits["next_token"]
        residual_embedding = output.value["residual_embedding"].tensor

        # loss
        losses["token"] = self.token_loss(
            prediction=t_logits, target=t_targets, mask=mask, mask_weights=mask_weights
        )
        losses["concept"] = self.concept_loss(
            concept_logits=c_logits, doc_spans=doc_spans, mask=mask
        )
        losses["reconstruction"] = self.reconstruction_loss(
            prediction=unknown_embedding, target=unknown_target_embedding, mask=mask
        )
        losses["independence_unknown"] = self.independence_unknown_loss(
            known_repr=known_embedding, other_repr=unknown_embedding
        )
        losses["independence_unknown"] = self.independence_residual_loss(
            known_repr=known_embedding, other_repr=residual_embedding
        )

        # steering
        if self.training and kwargs["step"] % self.steering_every_n_steps == 0:
            concept_ids = intervention_ids.reshape(-1).long()  # shape: [B]
            known_embeddings = self.bottleneck["known_embeddings"]
            table = (
                known_embeddings.K
                if known_embeddings.rank is None
                else known_embeddings.A @ known_embeddings.B
            )
            directions = F.normalize(
                table[concept_ids], dim=-1
            )  # shape: [B, D], each row's own concept direction
            gamma = steering.calibrate_gamma(
                directions, self._head, self.tau
            )  # shape: [B]

            with steering.injected_at(
                self, directions, gamma, batch["position_mask"], self.inj_layer
            ):
                steer_x, _ = self.prepare_input(batch)
                steer_out = self.inference.query(
                    query=["concepts", "next_token"], evidence={"input": steer_x}
                )

            losses["steering"] = self.steering_loss(
                next_token_logits=steer_out.logits["next_token"],
                lifted_token_ids=lifted_token_ids,
                position_mask=mask,
            )
            losses["respond"] = self.respond_loss(
                concept_logits=steer_out.logits["concepts"],
                concept_ids=intervention_ids,
                position_mask=mask,
            )

        # metrics
        if not self.training:
            # only to diagnose the model: compute causal effect of intervening on concepts
            output_1 = self.inference.query(
                query=["next_token"],
                evidence={"concepts": intervened_labels_1, "latent": latent},
            )
            output_0 = self.inference.query(
                query=["next_token"],
                evidence={"concepts": intervened_labels_0, "latent": latent},
            )
            logits_1, logits_0 = (
                output_1.logits["next_token"],
                output_0.logits["next_token"],
            )

            self.causal_concept_effect_metric.update(
                logits_1=logits_1, logits_0=logits_0
            )
            self.token_metric.update(prediction=t_logits, target=t_targets)
            self.concept_metric.update(concept_logits=c_logits, doc_spans=doc_spans)

        return {k: v for k, v in losses.items() if v is not None}
