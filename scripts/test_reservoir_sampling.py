"""Standalone smoke test for BabyAtlas's streaming reservoir sampler.

Run from the repository root:
    python scripts/test_reservoir_sampling.py

To sample a real corpus without downloading or running a tagging model:
    python scripts/test_reservoir_sampling.py \\
        --input-path experiments/data/raw/tinystories.txt --num-documents 5000
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from babysteerling.data.babyatlas.tag_chunks import _iter_documents, load_documents


DELIMITER = "<|endoftext|>"
DOCUMENTS = ["first story", "second story", "third story"]


def run_smoke_test():
    with TemporaryDirectory() as directory:
        corpus_path = Path(directory) / "stories.txt"
        corpus_path.write_text(DELIMITER.join(DOCUMENTS) + DELIMITER, encoding="utf-8")

        # A tiny read size forces at least one delimiter to span two reads.
        assert list(_iter_documents(corpus_path, DELIMITER, read_size=5)) == DOCUMENTS

        # Asking for more documents than exist retains all of them.
        assert set(load_documents(corpus_path, 10, DELIMITER)) == set(DOCUMENTS)

        # Reservoir sampling is deterministic for a fixed seed.
        sample_one = load_documents(corpus_path, 2, DELIMITER, seed=1337)
        sample_two = load_documents(corpus_path, 2, DELIMITER, seed=1337)
        assert sample_one == sample_two
        assert len(sample_one) == 2

    print("Reservoir sampling smoke test passed.")


def sample_real_corpus(input_path, num_documents, seed):
    """Run the sampler over a real corpus and display one sampled document."""
    documents = load_documents(input_path, num_documents, DELIMITER, seed)
    print(f"Sampled {len(documents)} documents from {input_path}.")
    if documents:
        print(f"First sampled document: {len(documents[0])} characters")
        print("\n--- First 500 characters ---")
        print(documents[0][:500])


def main():
    parser = argparse.ArgumentParser(description="Test BabyAtlas reservoir sampling.")
    parser.add_argument(
        "--input-path",
        type=Path,
        help="Optional real corpus to sample. Without this, run the fast synthetic smoke test.",
    )
    parser.add_argument("--num-documents", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.input_path is None:
        run_smoke_test()
    elif not args.input_path.is_file():
        parser.error(f"input file does not exist: {args.input_path}")
    else:
        sample_real_corpus(args.input_path, args.num_documents, args.seed)


if __name__ == "__main__":
    main()
