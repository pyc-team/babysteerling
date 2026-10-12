from typing import Type

import plotly.graph_objects as go
import torch
from torch import nn as nn
from torch.distributions import OneHotCategorical, Bernoulli
from torch.nn import Linear
from torch_concepts import ConceptVariable, EmbeddingVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD,
    BayesianNetwork,
    LinearEmbeddingToConcept,
    BaseInference,
    DeterministicInference,
)

from ..backbone.lm import ILM
from ..loss import NullLoss
from ..metric import NullMetric
from .utils import _out_concepts_and_n
from ..nn.encoder import ConceptToLowRankEmbeddings


class ConceptResidualMixin(ILM):
    """Concept bottleneck plus SIM residual"""

    def __init__(
        self,
        out_concepts,
        inference: Type[BaseInference],
        steering_every_n_steps: int = 10,
        inj_layer: int = 1,
        tau: float = 4.0,
        inference_kwargs: dict | None = None,
        token_loss: nn.Module | None = None,
        concept_loss: nn.Module | None = None,
        token_residual_loss: nn.Module | None = None,
        token_metric: nn.Module | None = None,
        token_residual_metric: nn.Module | None = None,
        concept_metric: nn.Module | None = None,
        causal_concept_effect_metric: nn.Module | None = None,
        causal_concept_effect_concept_metric: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(
            inference=inference, inference_kwargs=inference_kwargs, **kwargs
        )
        out_concepts, n_concepts = _out_concepts_and_n(out_concepts)
        self.steering_every_n_steps = steering_every_n_steps
        self.inj_layer = inj_layer
        self.tau = tau

        self.token_loss = token_loss or NullLoss()
        self.concept_loss = concept_loss or NullLoss()
        self.token_residual_loss = token_residual_loss or NullLoss()
        self.token_metric = token_metric or NullMetric()
        self.token_residual_metric = token_residual_metric or NullMetric()
        self.concept_metric = concept_metric or NullMetric()
        self.causal_concept_effect_metric = causal_concept_effect_metric or NullMetric()
        self.causal_concept_effect_concept_metric = (
            causal_concept_effect_concept_metric or NullMetric()
        )

        self._bottleneck = nn.ModuleDict(
            {
                "concepts": LinearEmbeddingToConcept(self.n_embed, n_concepts),
                "concept_embeddings": ConceptToLowRankEmbeddings(
                    n_concepts, self.n_embed
                ),
                "residual": Linear(self.n_embed + self.n_embed, self.n_embed),
            }
        )
        self._head = Linear(self.n_embed, self.vocab_size)

        self.concepts = ConceptVariable(
            "concepts", distribution=Bernoulli, size=1, members=list(out_concepts)
        )
        self.concept_embeddings = EmbeddingVariable(
            "concept_embeddings", distribution=Delta, size=self.n_embed
        )
        self.residual = EmbeddingVariable(
            "residual", distribution=Delta, size=self.n_embed
        )
        self.next_token = ConceptVariable(
            "next_token",
            distribution=OneHotCategorical,
            size=1,
            members=[f"token_{i}" for i in range(self.vocab_size)],
        )
        self.next_token_residual = ConceptVariable(
            "next_token_residual",
            distribution=OneHotCategorical,
            size=self.vocab_size,
        )

        self.concepts_cpd = ParametricCPD(
            self.concepts,
            parametrization={"logits": self._bottleneck["concepts"]},
            parents=[self.latent_var],
        )
        self.concept_embeddings_cpd = ParametricCPD(
            self.concept_embeddings,
            parametrization=self._bottleneck["concept_embeddings"],
            parents=[self.concepts],
        )
        self.residual_cpd = ParametricCPD(
            self.residual,
            parametrization=self._bottleneck["residual"],
            parents=[self.latent_var, self.concept_embeddings],
        )
        self.next_token_cpd = ParametricCPD(
            self.next_token,
            parametrization={"logits": self._head},
            parents=[self.concept_embeddings],
        )
        self.next_token_residual_cpd = ParametricCPD(
            self.next_token_residual,
            parametrization={"logits": self._head},
            parents=[self.residual],
        )

        self.pgm = BayesianNetwork(
            [
                self.input_var,
                self.latent_var,
                self.concepts,
                self.concept_embeddings,
                self.residual,
                self.next_token,
                self.next_token_residual,
            ],
            [
                self.input_cpd,
                self.latent_cpd,
                self.concepts_cpd,
                self.concept_embeddings_cpd,
                self.residual_cpd,
                self.next_token_cpd,
                self.next_token_residual_cpd,
            ],
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
            "token_residual": self.token_residual_metric,
            "concept": self.concept_metric,
            "causal_concept_effect": self.causal_concept_effect_metric,
            "causal_concept_effect_concept": self.causal_concept_effect_concept_metric,
        }

    def step(self, batch: dict, *args, **kwargs) -> dict[str, torch.Tensor]:
        x, extras = self.prepare_input(batch)
        mask = extras.get("mask")
        mask_weights = extras.get("p_mask")
        doc_spans = batch.get("doc_spans")
        t_targets = batch.get("targets")
        intervention_ids = batch.get("random_intervention_ids")
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
            query=["concepts", "latent"], evidence={"input": x}
        )
        c_logits, latent = (
            output.logits["concepts"].tensor,
            output.value["latent"].tensor,
        )
        c_probs = (
            torch.sigmoid(c_logits) if not self.training else known_labels
        )  # teacher forcing
        output = self.inference.query(
            query=["next_token"], evidence={"concepts": c_probs}
        )
        t_logits = output.logits["next_token"]
        output = self.inference.query(
            query=["next_token_residual"],
            evidence={"concepts": c_probs, "latent": latent},
        )
        t_logits_residual = output.logits["next_token_residual"]

        # loss
        losses["concept"] = self.concept_loss(
            concept_logits=c_logits, doc_spans=doc_spans, mask=mask
        )
        losses["token"] = self.token_loss(
            prediction=t_logits, target=t_targets, mask=mask, mask_weights=mask_weights
        )
        losses["token_residual"] = self.token_residual_loss(
            prediction=t_logits.detach() + t_logits_residual,
            target=t_targets,
            mask=mask,
            mask_weights=mask_weights,
        )

        # metrics
        if not self.training:
            # only to diagnose the model: compute causal effect of intervening on concepts
            # first using the residual
            output_1 = self.inference.query(
                query=["next_token_residual"],
                evidence={"concepts": intervened_labels_1, "latent": latent},
            )
            output_0 = self.inference.query(
                query=["next_token_residual"],
                evidence={"concepts": intervened_labels_0, "latent": latent},
            )
            logits_1, logits_0 = (
                output_1.logits["next_token_residual"],
                output_0.logits["next_token_residual"],
            )
            self.causal_concept_effect_metric.update(
                logits_1=logits_1, logits_0=logits_0
            )

            # then using concepts alone
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
            self.causal_concept_effect_concept_metric.update(
                logits_1=logits_1, logits_0=logits_0
            )

            self.concept_metric.update(concept_logits=c_logits, doc_spans=doc_spans)
            self.token_metric.update(prediction=t_logits, target=t_targets)
            self.token_residual_metric.update(
                prediction=t_logits + t_logits_residual, target=t_targets
            )

        return {k: v for k, v in losses.items() if v is not None}

    def diagnostics(self, decode_fn, *args, **kwargs):
        if self.bottleneck["concept_embeddings"].rank is not None:
            known_embeddings = (
                self.bottleneck["concept_embeddings"].A
                @ self.bottleneck["concept_embeddings"].B
            )
        else:
            known_embeddings = self.bottleneck["concept_embeddings"].K

        concept_weights = (
            known_embeddings @ self._head.weight.T
        )  # shape: [n_concepts, vocab_size]
        topk = 10
        top_token_ids = concept_weights.argsort(descending=True)[
            :, :topk
        ]  # shape: [n_concepts, topk]
        top_weights = concept_weights.gather(
            1, top_token_ids
        ).tolist()  # shape: [n_concepts, topk]
        top_token_names = [
            [decode_fn([token_id]) for token_id in token_ids]
            for token_ids in top_token_ids.tolist()
        ]  # shape: [n_concepts, topk]

        heatmap = go.Figure(
            go.Heatmap(
                z=top_weights,
                text=top_token_names,
                texttemplate="%{text}",
                y=self.concepts.members,
                x=[f"#{i + 1}" for i in range(topk)],
                colorscale="Viridis",
                colorbar=dict(title="weight"),
            )
        )
        heatmap.update_layout(
            title="Top-k tokens per concept",
            yaxis=dict(autorange="reversed"),
        )

        return {"concept_topk_heatmap": heatmap}
