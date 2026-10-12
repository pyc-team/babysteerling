# Baby Steerling Lab

The [Hydra](https://hydra.cc) + [Weights & Biases](https://wandb.ai) configurable environment to run experiments.
`run_auto.py` and `build_dataset_auto.py` just unpack a Hydra config and assemble data, model, and training pipelines.
The other scripts (`1_data.py`, `2_forward.py`, `3_trainer.py`, `4_pyc_style_model.py`) 
are simple, minimal examples to understand each component of `babysteerling`.

## Setup

From the repo root:
```bash
pip install -e ".[atlas]"   # editable install of babysteerling, incl. dataset-building deps
wandb login                 # one-time; skip if you plan to use mode=offline (see below)
```

## Building a dataset

Before training for the first time you need to build a concept-labeled dataset. 
The default config builds a small version of TinyStories:
```bash
cd lab
python build_dataset_auto.py
```
This downloads a raw corpus, trains a tokenizer on it, then runs the four Atlas stages (tag,
cluster, assign, tokenize) in order, writing everything to `atlas.output_dir` (default `./data`).
Every stage is idempotent: it's skipped if already done, so re-running after a partial failure,
or after changing a later stage's hyperparameter, doesn't redo finished work.

You can override data-generation hyperparameters as follows:
```bash
python build_dataset_auto.py atlas.num_documents=2000 atlas.k=80
```
`run_auto.py`'s default config already points `data.data_dir` at this same `./data`, so once the
build finishes, `python run_auto.py` just works.

**Which corpus gets built and how it's processed are two separate configs.**
`configs/data/corpus/tinystories.yaml` says what/where the data is (`sources`, a list of
`{name, url, document_delimiter}` entries, plus the shared `tokenizer_vocab_size` and
`boundary_token`) and how training loads it (`data_dir`, `block_size`, `batch_size`, ...).
These two are always edited/swapped together per-corpus. `configs/data/atlas.yaml` says how the
corpus is *processed*, shared across every corpus: sample size, which models to use,
clustering/dedup thresholds, and the two LLM prompt templates for tagging and labeling concepts.

To build from a different corpus, copy `configs/data/corpus/tinystories.yaml` to a new file (e.g.
`configs/data/corpus/my_corpus.yaml`) and replace its `sources`/loader entries. Then run:
```bash
python build_dataset_auto.py data/corpus=my_corpus
```
`configs/data/atlas.yaml` is a single shared file (not a swappable group) since every corpus here
is short children's stories; if a new corpus needs different tag/label prompts, turn it back into
a group (`configs/data/atlas/default.yaml` + `configs/data/atlas/my_corpus.yaml`) and reference it
the same way.

**To train on several corpora at once**, add more entries to `sources`:
```yaml
sources:
  - name: tinystories
    url: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
    document_delimiter: "<|endoftext|>"
  - name: my_other_corpus
    url: https://example.com/my_other_corpus.txt
    document_delimiter: "\n\n\n"   # can differ per source
```
Each source is downloaded and tagged independently, but all sources' tags are combined into one
shared concept library, and every source is assigned concepts from that same library. This is
what makes it a real union: a concept means the same thing no matter which source a chunk came
from, rather than just concatenating unrelated datasets. `boundary_token` (default
`<|endoftext|>`) is separate from `document_delimiter`: it's the single marker the pipeline
inserts after every chunk once tokenized, and it doesn't need to match any source's delimiter.

## Running a single experiment

To train a default model with default hyperparameters, just run:
```bash
cd lab
python run_auto.py
```
This uses the defaults in `configs/config.yaml`.

Override any hyperparameter from the command line, no file editing needed:
```bash
python run_auto.py training.lr=1e-3 training.max_steps=2000
python run_auto.py model/backbone=dlm model/bottleneck=residual
```

Each run saves a checkpoint to `./checkpoints/babysteerling-tests/<run-id>/last.ckpt`, and
logs a short sample generation to W&B at the end.

## Config layout

`configs/` has three buckets: `data/` (what the corpus is and how it's processed/loaded),
`model/` (architecture and loss), `experiment/` (run mechanics -- W&B, training schedule,
sweeps). Within each, a group is a real subfolder only where there's an actual choice to make
(`data/corpus/`, `model/backbone/`, `model/bottleneck/`); everything else is a single flat file,
tuned via field overrides (e.g. `training.lr=1e-3`) rather than swapped as a whole preset.

Backbone and bottleneck are independent groups; each bottleneck carries its own matching loss
(see `configs/model/bottleneck/*.yaml` -- loss isn't a separate group, since most loss/bottleneck
pairings are meaningless), so a combo is always just two flags:
```bash
python run_auto.py model/backbone=dlm model/bottleneck=residual
```
The one real fork is the `cb` bottleneck: `model/bottleneck=cb` is a no-concept-loss ablation,
`model/bottleneck=cb_with_loss` is the same bottleneck with real concept supervision added.

## Running experiments in parallel (sweeps)

To run multiple experiments in parallel execute
```bash
bash run_sweep.sh
```
This runs `configs/experiment/sweep_all.yaml`, which sweeps the full backbone x bottleneck
cartesian product. To add a new backbone or bottleneck to the sweep, add its config under
`configs/model/backbone/` or `configs/model/bottleneck/` and add it to `sweep_all.yaml`'s
`hydra.sweeper.params` lists -- no per-combo file needed.
