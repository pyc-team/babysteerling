"""Stage 1: sample documents from a raw text corpus and LLM-tag each with free-form concept tags.

The high-recall step: cast a wide net of raw, possibly redundant tags per document. Stage 2
(build_concepts) turns this noisy tag pool into a clean concept library, so overlapping or
overly specific tags here are fine, expected even.

Assumes one text file, documents separated by a delimiter (e.g. TinyStories's "<|endoftext|>"),
short enough that one document = one chunk. Corpus, delimiter, and prompt are all parameters;
see experiments/configs/corpus/ for how a Hydra caller picks them.
"""

import json
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT_TEMPLATE = (
    "Read the short children's story below and list 5 to 10 free-form tags describing it: "
    "its characters, setting, theme or moral, plot events, and tone. "
    "Respond with ONLY a JSON array of short lowercase tag strings, no other text.\n\n"
    "Story:\n{document}\n\nTags:"
)


def _iter_documents(input_path, delimiter, read_size=1024 * 1024):
    """Read documents one at a time without loading the whole corpus into RAM."""
    if not delimiter:
        raise ValueError("document delimiter must be a non-empty string")

    # Keep the final piece in case it continues in the next chunk.
    with open(input_path, 'r', encoding='utf-8') as f:
        remainder = ""
        while chunk := f.read(read_size):
            pieces = (remainder + chunk).split(delimiter)
            remainder = pieces.pop()
            for piece in pieces:
                document = piece.strip()
                if document:
                    yield document

        document = remainder.strip()
        if document:
            yield document


def load_documents(input_path, num_documents, delimiter="<|endoftext|>", seed=1337):
    """Sample documents uniformly while keeping only ``num_documents`` in memory."""
    if num_documents < 0:
        raise ValueError("num_documents must be non-negative")
    if num_documents == 0:
        return []

    rng = random.Random(seed)
    reservoir = []
    documents_seen = 0
    for document in _iter_documents(input_path, delimiter):
        documents_seen += 1
        if len(reservoir) < num_documents:
            reservoir.append(document)
            continue

        # Pick a position among all documents seen so far. Replace only if it is in
        # the reservoir, giving every document an equal chance of being selected.
        replacement_index = rng.randrange(documents_seen)
        if replacement_index < num_documents:
            reservoir[replacement_index] = document

    # Randomize the sampled order reproducibly.
    rng.shuffle(reservoir)
    return reservoir


def parse_tags(output_text):
    """Pull the JSON array of tags out of a (possibly noisy) LLM completion."""
    match = re.search(r"\[.*\]", output_text, re.DOTALL)
    if not match:
        return None
    try:
        tags = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        return None
    tags = [t.strip().lower() for t in tags if t.strip()]
    return tags or None


def tag_chunks(
    input_path,
    output_path,
    num_documents=5000,
    document_delimiter="<|endoftext|>",
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    batch_size=16,
    max_new_tokens=150,
    seed=1337,
    device=None,
):
    """Samples `num_documents` documents from `input_path`, LLM-tags each with free-form concept
    tags, and writes one JSON line per successfully-tagged document to `output_path`.

    `prompt_template` must contain a `{document}` placeholder; override it for corpora that
    don't fit the default "children's story" framing.

    Idempotent: skips if `output_path` already exists, so re-running doesn't redo LLM inference.
    """
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping tagging.")
        return

    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    print("Loading documents...")
    documents = load_documents(
        input_path, num_documents, delimiter=document_delimiter, seed=seed
    )
    print(f"Sampled {len(documents)} documents.")

    print(f"Loading {model_name} on {device}...")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    # left padding so every sequence in a batch ends at the same position,
    # letting us slice out just the generated continuation below
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    num_failed = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start : batch_start + batch_size]
            # build one chat-formatted prompt per document, batched for throughput
            prompts = [
                tok.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": prompt_template.format(document=document),
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for document in batch
            ]
            inputs = tok(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )

            # strip the (left-padded) prompt tokens, keep only the generated continuation
            new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
            completions = tok.batch_decode(new_tokens, skip_special_tokens=True)

            for i, (document, completion) in enumerate(zip(batch, completions)):
                tags = parse_tags(completion)
                if tags is None:
                    num_failed += 1
                    continue
                chunk_id = batch_start + i
                out_f.write(
                    json.dumps({"chunk_id": chunk_id, "text": document, "tags": tags})
                    + "\n"
                )

            print(
                f"Tagged {min(batch_start + batch_size, len(documents))}/{len(documents)} documents "
                f"({num_failed} failed so far)"
            )

    print(
        f"Done. {num_failed}/{len(documents)} documents failed to parse and were skipped."
    )
    print(f"Wrote tags to {output_path}")
