from torch import nn
from .encoder import SparseEmbeddingToConcept


class ResidualModule(nn.Module):
    """epsilon = h - k_hat - u_hat: whatever the two concept heads fail to reconstruct.

    Dropout on epsilon discourages the model from routing information through this
    uninterpretable channel just because it's easier than going through a concept -- it should
    only carry what genuinely can't be expressed as a combination of concepts.
    """

    def __init__(self, p_epsilon=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p_epsilon)

    def forward(self, h, k_hat, u_hat):
        epsilon = h - k_hat - u_hat  # shape: [B, T, d], same shape as h
        return self.dropout(epsilon)


class ConceptBottleneck(nn.Module):
    """Composes the three heads: h_bar = k_hat + u_hat + epsilon."""

    def __init__(self, d, n, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None):
        super().__init__()
        self.n = n
        self.m = unknown_ratio * n
        self.known = SparseEmbeddingToConcept(d, n, top_k=top_k_known)
        self.unknown = SparseEmbeddingToConcept(d, self.m, rank=unknown_rank, top_k=top_k_unknown)
        self.residual = ResidualModule(p_epsilon)

    def forward(self, h, known_labels=None):
        k_logits = self.known.g(h)
        k, k_hat = self.known(h, logits=k_logits)
        u, u_hat = self.unknown(h.detach())

        k_hat_gt, u_hat_gt = None, None
        if known_labels is not None:
            k_hat_gt = self.known.ground_truth_embedding(known_labels)  # shape: [B, T, d]
            u_hat_gt = h.detach() - k_hat_gt  # shape: [B, T, d], target for the unknown head's reconstruction loss

        epsilon = self.residual(h, k_hat, u_hat)  # shape: [B, T, d]
        h_bar = k_hat + u_hat + epsilon  # shape: [B, T, d], exactly reconstructs h in expectation

        intermediates = {
            'k_logits': k_logits, 'k': k, 'u': u, 'k_hat': k_hat, 'u_hat': u_hat,
            'k_hat_gt': k_hat_gt, 'u_hat_gt': u_hat_gt, 'epsilon': epsilon,
        }
        return h_bar, intermediates
