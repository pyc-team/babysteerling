import pytest
import torch
from torch_concepts.nn import DeterministicInference

from babysteerling.loss import (
    ConceptLoss,
    ExpressLoss,
    IndependenceLoss,
    ReconstructionLoss,
    RespondLoss,
    TokenLoss,
    Weighted,
)
from babysteerling.metric import (
    CausalConceptEffect,
    ConceptAUC,
    TokenAccuracy,
)
from babysteerling.factory import BACKBONES, BOTTLENECKS, build_model

VOCAB, BLOCK, N_CONCEPTS, N_EMBED = 51, 8, 4, 16
MASK_TOKEN_ID = VOCAB - 1


def make_batch(batch_size: int) -> dict:
    return {
        "input_ids": torch.randint(0, MASK_TOKEN_ID, (batch_size, BLOCK)),
        "targets": torch.randint(0, MASK_TOKEN_ID, (batch_size, BLOCK)),
        "known_labels": torch.randint(0, 2, (batch_size, BLOCK, N_CONCEPTS)).float(),
        "random_intervention_ids": torch.randint(0, N_CONCEPTS, (batch_size, 1, 1)),
        "position_mask": torch.zeros(batch_size, BLOCK, dtype=torch.bool),
        "lifted_tokens_intervention_ids": torch.randint(
            0, MASK_TOKEN_ID, (batch_size, 1, 5)
        ),
    }


def all_loss_metric_kwargs() -> dict:
    return dict(
        token_loss=Weighted(TokenLoss(), 1.0),
        concept_loss=Weighted(ConceptLoss(), 1.0),
        token_residual_loss=Weighted(TokenLoss(), 1.0),
        reconstruction_loss=Weighted(ReconstructionLoss(), 1.0),
        independence_unknown_loss=Weighted(IndependenceLoss(), 1.0),
        independence_reconstruction_loss=Weighted(IndependenceLoss(), 1.0),
        steering_loss=Weighted(ExpressLoss(), 1.0),
        respond_loss=Weighted(RespondLoss(), 1.0),
        token_metric=TokenAccuracy(),
        token_residual_metric=TokenAccuracy(),
        concept_metric=ConceptAUC(),
        causal_concept_effect_metric=CausalConceptEffect(),
        causal_concept_effect_concept_metric=CausalConceptEffect(),
    )


COMBINATIONS = [
    (backbone, bottleneck)
    for backbone in BACKBONES
    for bottleneck in ["none"] + list(BOTTLENECKS)
]


@pytest.mark.parametrize(
    "backbone,bottleneck", COMBINATIONS, ids=[f"{b}-{k}" for b, k in COMBINATIONS]
)
def test_combination_builds_and_runs(backbone, bottleneck):
    is_diffusion = backbone == "dlm"
    kwargs = dict(
        vocab_size=VOCAB,
        block_size=BLOCK,
        n_embed=N_EMBED,
        num_heads=2,
        num_kv_heads=1,
        n_layers=1,
        dropout=0.0,
        inference=DeterministicInference,
    )
    if bottleneck != "none":
        kwargs["out_concepts"] = N_CONCEPTS
    if is_diffusion:
        kwargs["mask_token_id"] = MASK_TOKEN_ID
        kwargs["diff_block_len"] = 4

    model = build_model(
        backbone=backbone, bottleneck=bottleneck, **kwargs, **all_loss_metric_kwargs()
    )
    batch = make_batch(batch_size=3)

    model.train()
    losses = model.step(batch, step=0)
    train_loss = sum(losses.values())
    assert torch.isfinite(train_loss)
    train_loss.backward()

    model.eval()
    with torch.no_grad():
        losses = model.step(batch, step=0)
        eval_loss = sum(losses.values())
    assert torch.isfinite(eval_loss)

    for metric in model.metrics.values():
        value = metric.compute()
        if value is None:
            continue
        values = value.values() if isinstance(value, dict) else [value]
        assert all(torch.isfinite(torch.as_tensor(v)) for v in values)

    generated = model.generate(batch_size=1, max_new_tokens=4)
    assert len(generated) > 0
