# Data Pipeline Implementation Checklist

This checklist turns the plan into ordered work items with sign-off gates.

## Goal

Build a Common Crawl pipeline that processes **1 TB** of input, does **exact deduplication first**, produces a versioned dataset artifact, and feeds the existing training -> eval -> publish -> serve loop.

## Phase 0: Lock the contract

- [ ] Confirm source: Common Crawl.
- [ ] Confirm first target scale: 1 TB input.
- [ ] Confirm dedup scope: exact dedup only for v1.
- [ ] Confirm storage shape: raw inputs, cleaned dataset, and manifests all versioned in S3.
- [ ] Confirm machine-readable linkage: dataset manifest -> training run -> checkpoint -> eval -> published model.

Gate:
- [ ] Reviewer can restate the scope in one paragraph without ambiguity.

## Phase 1: Raw ingestion

- [ ] Add a pipeline entrypoint, e.g. `data_pipeline/ingest_commoncrawl.py`.
- [ ] Fetch or stage raw Common Crawl inputs into S3 under a run root.
- [ ] Record raw source URI, retrieval time, and raw checksum.
- [ ] Emit a raw input manifest.

Gate:
- [ ] A raw input batch can be replayed from its manifest and checksum.

## Phase 2: Normalization and filtering

- [ ] Add a normalization step, e.g. `data_pipeline/normalize.py`.
- [ ] Strip obvious boilerplate, malformed records, and low-signal text.
- [ ] Record filter counts and rejection reasons.
- [ ] Keep the process deterministic.

Gate:
- [ ] The same input batch produces the same filtered output and the same filter stats.

## Phase 3: Exact deduplication

- [ ] Add exact dedup logic, e.g. `data_pipeline/dedup_exact.py`.
- [ ] Remove byte-identical or normalized-identical records.
- [ ] Record dedup counts and surviving corpus size.
- [ ] Do not imply near-dedup yet.

Gate:
- [ ] Dedup stats are reproducible and visible in the manifest.

## Phase 4: Tokenization and packing

- [ ] Add tokenization step, e.g. `data_pipeline/tokenize_pack.py`.
- [ ] Pin tokenizer version.
- [ ] Record max length and packing strategy.
- [ ] Emit token counts and packed sequence counts.

Gate:
- [ ] Packed dataset can be regenerated exactly from the manifest.

## Phase 5: Dataset artifact and manifest

- [ ] Write `dataset_manifest.json`.
- [ ] Store dataset version, source checksums, output checksum, byte counts, token counts, and build parameters.
- [ ] Publish dataset artifact to S3 under `s3://<bucket>/runs/<run-name>/datasets/<dataset-version>/`.
- [ ] Make the manifest machine-readable and auditable.

Gate:
- [ ] A reviewer can trace a dataset version back to its raw inputs and forward to training.

## Phase 6: Training integration

- [ ] Add a dataset selector to the training job scripts.
- [ ] Make Slurm training jobs consume a dataset version from S3.
- [ ] Record `dataset_version` in `training_metrics.json`.
- [ ] Keep the existing checkpoint/export/publish flow unchanged.

Gate:
- [ ] A training run can be reproduced from a dataset version alone.

## Phase 7: Evaluation and commercial reporting

- [ ] Compare the dataset-backed run against the toy baseline.
- [ ] Record GPU hours, tokens/sec, eval result, and cost per useful model.
- [ ] Add the dataset version to the commercial report output.
- [ ] Link the report back to the dataset manifest and training checkpoint.

Gate:
- [ ] The report shows a measurable before/after story.

## Phase 8: Scale proof

- [ ] Run the pipeline on 1 TB of Common Crawl.
- [ ] Confirm it completes without manual intervention.
- [ ] Confirm the manifest and output are still traceable and reproducible.

Gate:
- [ ] The pipeline survives a scale that is operationally real, not a demo size.

## First files to add

- `data_pipeline/ingest_commoncrawl.py`
- `data_pipeline/normalize.py`
- `data_pipeline/dedup_exact.py`
- `data_pipeline/tokenize_pack.py`
- `data_pipeline/write_manifest.py`
- `slurm/data-pipeline.sbatch`
- `docs/data-pipeline-manifest-format.md`

## Done means

You can say all of the following without hand-waving:

- this dataset came from Common Crawl
- this version was normalized, deduplicated, tokenized, and packed in a deterministic way
- this manifest links to the training run, checkpoint, eval result, and published model
- this pipeline ran at 1 TB scale and held up
