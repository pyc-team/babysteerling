import math
import unittest

import torch
from torch.nn import functional as F

from babysteerling.loss import ConceptLoss, ReconstructionLoss
from babysteerling.model import build_model
from babysteerling.nn.bottleneck import ConceptBottleneck


class ConceptLossTest(unittest.TestCase):
    def test_logit_loss_matches_probability_soft_or(self):
        k_logits = torch.tensor(
            [[[-1.0, 0.5, -0.2], [0.3, -0.7, 1.1], [-0.4, 0.2, -1.2]]],
            dtype=torch.float64,
        )
        doc_spans = [(0, 0, 3, [0, 2])]

        actual = ConceptLoss()(k_logits, doc_spans)
        k_chunk = 1 - torch.prod(1 - torch.sigmoid(k_logits[0]), dim=0)
        expected = F.binary_cross_entropy(
            k_chunk, torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        )

        torch.testing.assert_close(actual, expected)

    def test_logit_loss_is_finite_for_saturated_predictions(self):
        k_logits = torch.tensor([[[-80.0, 80.0]] * 256], requires_grad=True)

        loss = ConceptLoss()(k_logits, [(0, 0, 256, [0])])
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(k_logits.grad).all())


class GradientRoutingTest(unittest.TestCase):
    def test_reconstruction_trains_heads_but_not_backbone_input(self):
        torch.manual_seed(0)
        bottleneck = ConceptBottleneck(d=4, n=3, unknown_ratio=2, p_epsilon=0.0)
        h = torch.randn(2, 3, 4, requires_grad=True)
        known_labels = torch.zeros(2, 3, 3)
        known_labels[..., 0] = 1

        _, intermediates = bottleneck(h, known_labels=known_labels)
        loss = ReconstructionLoss()(
            intermediates["u_hat"], intermediates["u_hat_gt"],
        )
        loss.backward()

        self.assertIsNone(h.grad)
        self.assertGreater(bottleneck.known.K.grad.abs().sum(), 0)
        self.assertGreater(bottleneck.unknown.K.grad.abs().sum(), 0)
        self.assertGreater(bottleneck.unknown.g[0].weight.grad.abs().sum(), 0)

    def test_concept_bias_matches_soft_or_window_scale(self):
        model = build_model(
            vocab_size=32,
            n_concepts=3,
            block_size=16,
            n_embed=8,
            num_heads=2,
            num_kv_heads=1,
            n_layers=1,
            dropout=0.0,
        )

        expected = torch.full_like(
            model.bottleneck.known.g[-1].bias, -math.log(16),
        )
        torch.testing.assert_close(model.bottleneck.known.g[-1].bias, expected)


if __name__ == "__main__":
    unittest.main()
