from typing import Type, Union

import plotly.graph_objects as go
import torch
from torch import nn as nn
from torch.distributions import OneHotCategorical, Bernoulli
from torch.nn import Linear
import torch_concepts as pyc
from torch_concepts import ConceptVariable, Annotations, EmbeddingVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD,
    BayesianNetwork,
    BaseInference,
)

from .alm import ALM
from .lm import ILM
from ..loss import LossOutput
from ..nn.encoder import ConceptToLowRankEmbeddings


class ConceptResidualALM(ALM, ILM):
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
        inj_layer: int = 1,
        tau: float = 4.0,
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
        self.inj_layer = inj_layer
        self.tau = tau

        self._bottleneck = nn.ModuleDict(
            {
                "concepts": pyc.nn.LinearEmbeddingToConcept(n_embed, n_concepts),
                "concept_embeddings": ConceptToLowRankEmbeddings(n_concepts, n_embed),
                "residual": Linear(n_embed + n_embed, n_embed),
            }
        )
        self._head = Linear(self.n_embed, self.vocab_size)

        self.concepts = ConceptVariable(
            "concepts", distribution=Bernoulli, size=1, members=list(out_concepts)
        )
        self.concept_embeddings = EmbeddingVariable(
            "concept_embeddings", distribution=Delta, size=n_embed
        )
        self.residual = EmbeddingVariable("residual", distribution=Delta, size=n_embed)
        self.next_token = ConceptVariable(
            "next_token",
            distribution=OneHotCategorical,
            size=1,
            members=[f"token_{i}" for i in range(vocab_size)],
        )
        self.next_token_residual = EmbeddingVariable(
            "next_token_residual",
            distribution=Delta,
            size=vocab_size,
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
            parametrization=self._head,
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

    def step(self, batch: dict, *args, **kwargs) -> LossOutput:
        x = self.tokens_to_embedding(batch["input_ids"])

        out_concepts = self.inference.query(
            query=["concepts", "latent"],
            evidence={"input": x},
        )

        concepts = torch.sigmoid(out_concepts.logits["concepts"].tensor)
        if self.training:
            concepts = batch["known_labels"]  # teacher forcing

        out_next_token = self.inference.query(
            query=["next_token"],
            evidence={"concepts": concepts},
        )
        out_next_token_residual = self.inference.query(
            query=["next_token_residual"],
            evidence={
                "concepts": concepts,
                "latent": out_concepts.value["latent"].tensor,
            },
        )

        total_loss = self.loss_fn(out_concepts, batch, ["concept"], ["concept_auc"])
        total_loss += self.loss_fn(out_next_token, batch, ["token"], ["token_accuracy"])
        total_loss += self.loss_fn(
            {
                "next_token": out_next_token,
                "next_token_residual": out_next_token_residual,
            },
            batch,
            ["token_residual"],
            ["token_accuracy_residual"],
        )

        # only to diagnose the model: compute causal effect of intervening on concepts
        B, T, C = batch["known_labels"].shape
        device = batch["random_intervention_ids"].device
        intervention_mask = batch["random_intervention_ids"] == torch.arange(
            C, device=device
        )
        intervened_labels_1 = torch.where(intervention_mask, 1.0, batch["known_labels"])
        intervened_labels_0 = torch.where(intervention_mask, 0.0, batch["known_labels"])

        output_1 = self.inference.query(
            query=["next_token_residual"],
            evidence={
                "concepts": intervened_labels_1,
                "latent": out_concepts.value["latent"].tensor,
            },
        )
        output_0 = self.inference.query(
            query=["next_token_residual"],
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

        output_1 = self.inference.query(
            query=["next_token"],
            evidence={"concepts": intervened_labels_1},
        )
        output_0 = self.inference.query(
            query=["next_token"],
            evidence={"concepts": intervened_labels_0},
        )
        total_loss += self.loss_fn(
            {"next_token_1": output_1, "next_token_0": output_0},
            batch,
            None,
            ["causal_concept_effect_concept"],
        )

        return total_loss

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
