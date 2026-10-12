from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from torch.nn import Identity
from torch_concepts import EmbeddingVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import ParametricCPD

from ..nn.backbone import TransformerModel, TokensToEmbeddings


class LM(nn.Module, ABC):
    """Language model interface."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int,
        num_heads: int,
        num_kv_heads: int,
        n_layers: int,
        dropout: float,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.tokens_to_embedding = TokensToEmbeddings(vocab_size, n_embed, block_size)
        self.backbone = TransformerModel(
            n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads, attn_mask
        )

        self.input_var = EmbeddingVariable("input", distribution=Delta, size=n_embed)
        self.latent_var = EmbeddingVariable("latent", distribution=Delta, size=n_embed)

        self.input_cpd = ParametricCPD(
            self.input_var, parametrization=Identity(), parents=[]
        )
        self.latent_cpd = ParametricCPD(
            self.latent_var, parametrization=self.backbone, parents=[self.input_var]
        )

    @property
    @abstractmethod
    def head(self) -> nn.Module:
        """Subclasses must return an nn.Module."""
        pass

    @abstractmethod
    def step(self, batch: dict, **kwargs) -> dict[str, torch.Tensor]:
        """Runs the model on a batch and returns a dict of named, already-weighted loss values
        (whichever of this model's `*_loss` attributes are configured -- see e.g. ALM/
        ConceptBottleneckMixin). Sum the dict for the total loss; each entry doubles as the
        per-term value to log."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Generate a sequence of tokens from the model."""
        raise NotImplementedError

    @property
    def metrics(self) -> dict:
        """Named stateful eval metrics (this model's `*_metric` attributes) -- for the trainer's
        epoch-end compute()/reset() loop. Subclasses override to include their own."""
        return {}

    def diagnostics(self, *args, **kwargs) -> dict:
        """Optional post-training checks a subclass can expose (e.g. concept activation
        stats, steering sanity checks). DiagnosticsLogger calls this and logs whatever it
        returns; no-op by default so models that don't define one aren't affected.
        """
        return {}


class ILM(LM, ABC):
    """Interface for interpretable language models (ILMs)."""

    @property
    @abstractmethod
    def bottleneck(self) -> nn.ModuleDict:
        """Subclasses must return an nn.ModuleDict that implements the bottleneck."""
        pass


class DLM(LM, ABC):
    """Interface for diffusion language models."""

    @property
    @abstractmethod
    def attn_mask(self) -> torch.Tensor | None:
        """Subclasses must return an attention mask Tensor or None."""
        pass
