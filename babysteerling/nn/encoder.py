import torch
import torch.nn as nn

from torch_concepts.nn import BaseConceptLayer


def sparsify_top_k(activations, k):
    """Zero out every activation except the top-k per token (optional, off by default).

    Intuition: forces each token to "explain itself" via a small number of active concepts
    instead of a dense mixture, which is closer to how a human would describe a piece of text
    (a few salient concepts, not a weighted blend of the entire library).
    """
    if k is None or k >= activations.shape[-1]:
        return activations
    top_vals, top_idx = torch.topk(activations, k, dim=-1)  # shape: [..., n] -> [..., k] (values and their indices)
    sparse = torch.zeros_like(activations)
    sparse.scatter_(-1, top_idx, top_vals)  # write the top-k values back into their original positions, rest stay 0
    return sparse


class SparseEmbeddingToConcept(BaseConceptLayer):
    """Concept encoder: Activation -> weighted-embedding mechanism as the known head, with an optional
    low-rank embedding table for when m is large enough that a dense [m, d] table would dominate the parameter count.
    """

    def __init__(self, in_embeddings, out_concepts, hidden_dim=None, rank=None, top_k=None):
        super().__init__(
            out_concepts=out_concepts,
            in_concepts=None,  # Concepts come from encoder, not traditional input
            in_embeddings=in_embeddings
        )
        m = self.out_concepts_shape
        d = self.in_embeddings_shape
        hidden_dim = hidden_dim or d
        self.g = nn.Sequential(nn.Linear(d, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, m))
        self.rank = rank
        if rank is None:
            self.K = nn.Parameter(torch.randn(m, d) * 0.02)  # shape: [m, d]
        else:
            # low-rank factorization U = A @ B: cuts params from m*d to rank*(m+d), and turns
            # the per-token [m, d] matmul into two smaller ones -- worthwhile once m >> rank
            self.A = nn.Parameter(torch.randn(m, rank) * 0.02)  # shape: [m, rank]
            self.B = nn.Parameter(torch.randn(rank, d) * 0.02)  # shape: [rank, d]
        self.top_k = top_k

    def activation(self, embeddings, logits=None):
        logits = self.g(embeddings) if logits is None else logits
        u = torch.sigmoid(logits)  # shape: [B, T, d] -> [B, T, m]
        return sparsify_top_k(u, self.top_k)

    def embed(self, u):
        if self.rank is None:
            return u @ self.K  # shape: [B, T, m] @ [m, d] -> [B, T, d]
        return (u @ self.A) @ self.B  # shape: [B, T, m] @ [m, rank] -> [B, T, rank] -> @ [rank, d] -> [B, T, d]

    def forward(self, embeddings, logits=None):
        # split into activation()/embed() (rather than one inline forward) so
        # babysteerling.steering's InterventionModule can wrap activation() alone -- a clean
        # single-tensor-in/out callable -- to intervene on concept activations without touching
        # the embedding-sum step
        u = self.activation(embeddings, logits=logits)
        u_hat = self.embed(u)
        return u, u_hat

    def ground_truth_embedding(self, known_labels):
        # weighted sum of K by the ground-truth chunk-level labels, broadcast to every token
        # position of the chunk -- this is the reconstruction loss's target for the unknown head
        if self.rank is None:
            return known_labels.float() @ self.K  # shape: [B, T, n] @ [n, d] -> [B, T, d]
        return (known_labels.float() @ self.A) @ self.B
