from torch_concepts.nn import DeterministicInference

from babysteerling.loader import ConceptDataModule
from babysteerling.loss import TokenLoss, ConceptLoss, Weighted
from babysteerling.metric import TokenAccuracy, ConceptAUC
from babysteerling.factory import build_model


def main():
    # 1. Initialize and set up the DataModule
    dm = ConceptDataModule(
        data_dir="./data",
        block_size=1024,
        batch_size=32,
        num_workers=4,
    )
    dm.setup("fit")

    # 2. Fetch a single batch from the train_dataloader
    batch = next(iter(dm.train_dataloader()))

    # 3. Inspect keys and tensor shapes
    print("--- Batch Keys ---")
    print(batch.keys())

    print("\n--- Tensor Shapes & Metadata ---")
    print(f"input_ids:    {batch['input_ids'].shape}   # [B, T]")
    print(f"targets:      {batch['targets'].shape}   # [B, T]")
    print(f"known_labels: {batch['known_labels'].shape} # [B, T, n_concepts]")
    print(f"doc_spans:    {len(batch['doc_spans'])} spans across batch")

    # 4. Inspect a sample doc_span tuple (batch_idx, tok_start, tok_end, concept_ids)
    if batch["doc_spans"]:
        print(f"First span:   {batch['doc_spans'][0]}")

    # 5. Instantiate the Model & Loss Function
    # Model hyperparams
    vocab_size = 2000
    block_size = dm.hparams.block_size
    n_embed = 128
    num_heads = 4
    num_kv_heads = 2
    n_layers = 4
    dropout = 0.2
    out_concepts = dm.n_concepts

    # 6. Instantiate Model -- loss/metric modules are passed straight in as named kwargs, same
    # as any other model hyperparameter
    inference_type = DeterministicInference
    model = build_model(
        backbone="alm",
        bottleneck="cb",
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        n_layers=n_layers,
        dropout=dropout,
        out_concepts=out_concepts,
        inference=inference_type,
        token_loss=Weighted(TokenLoss(), 1.0),
        concept_loss=Weighted(ConceptLoss(), 1.0),
        token_metric=TokenAccuracy(),
        concept_metric=ConceptAUC(),
    )

    # 7. Execute 1 Step via model.step(batch)
    model.eval()
    losses = model.step(batch, step=0)

    # 8. Inspect Results
    print("\n--- Model Output Inspection ---")
    print(f"Total Loss:            {sum(losses.values()).item():.4f}")

    print("\n--- Metrics Readout ---")
    for name, metric in model.metrics.items():
        value = metric.compute()
        if value is None:
            continue
        print(f"  {name:<20}: {value}")

    # 9. Backward Pass Verification
    sum(losses.values()).backward()
    print("\nBackward pass completed successfully!")


if __name__ == "__main__":
    main()
