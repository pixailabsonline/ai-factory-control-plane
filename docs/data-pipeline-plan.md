# Data Pipeline Plan

This project already proves the training and release path for GPU models:

- Slurm job submission
- FSDP training
- checkpointing and recovery
- eval gating
- export and publish
- vLLM serving

The next missing layer is the data pipeline that feeds training. The goal is to turn messy raw text into a training-ready corpus that is:

- traceable
- clean
- deduplicated
- reproducible
- versioned
- contamination-safe
- large enough to be operationally real

## Target Scope

Start with **Common Crawl** as the source and prove the pipeline on **1 TB** of input. That is large enough to be annoying in a way that exposes real operational failure modes, but still simple enough to build and explain.

For the first version, implement **exact deduplication** only. Near-deduplication with MinHash or SimHash is a sensible phase two, but it should not be implied by the first pass if it is not built.

## Where It Fits

The data pipeline sits **before** training and **feeds** the existing train/eval/publish flow.

```text
raw data -> filter -> dedupe -> tokenize -> pack -> version -> train -> checkpoint -> eval -> publish -> serve
```

This matters because the current repo proves that the cluster can run jobs, but not yet that it can consistently produce a better training corpus than a toy dataset.

## Commercial Value

The commercial story is not "we moved text around."

The commercial story is:

- we can reliably convert raw data into training-ready data
- we can prove what went into each model
- we can reduce wasted GPU hours by training on better data
- we can trace a passing checkpoint back to an exact dataset version

## What Needs To Be Built

### 1. Source provenance

Record:

- source name
- retrieval time
- license or policy basis
- raw artifact checksum
- original location

### 2. Cleaning and filtering

Implement filters for:

- spam
- boilerplate
- malformed records
- very low-signal text
- obvious junk

### 3. Deduplication

Remove:

- exact duplicates
- near duplicates only in a later phase

This avoids wasted tokens and improves the quality of the training mix.

### 4. Tokenization and packing

Convert cleaned text into fixed training sequences efficiently.

This step should record:

- tokenizer version
- max length
- packing strategy
- resulting token count

### 5. Dataset versioning

Every output dataset should be immutable and versioned.

Recommended shape:

- `s3://<bucket>/runs/<run-name>/datasets/<dataset-version>/`

Each dataset version should include a manifest with:

- source list
- filters applied
- dedupe stats
- token counts
- checksum
- build timestamp
- source checksum
- output dataset checksum
- dataset size in bytes
- input size in bytes
- upstream raw artifact location

The manifest should also link forward to the downstream training run and eval result once those exist:

- dataset version
- training run ID
- checkpoint path
- eval result path
- published model artifact path

### 6. Reproducibility

The same inputs and the same pipeline should produce the same dataset artifact.

That means:

- deterministic processing
- pinned tool versions
- recorded parameters
- stable manifests

### 7. Contamination control

Prevent benchmark leakage and accidental overlap between:

- training
- validation
- eval

The pipeline should track dataset splits explicitly and refuse to build if a forbidden source is detected.

## Operational Scale

The pipeline should be run at a scale that is annoying enough to matter.

- 1 GB proves the logic
- **1 TB of Common Crawl proves the system**

The point is not to build a toy ETL job. The point is to prove the pipeline stays stable when the input size is large enough to expose bad assumptions.

## Proposed Implementation Shape

Keep the implementation simple and visible:

- ingest raw sources into S3
- run a normalization job
- run dedupe and filtering
- tokenize and pack
- write a dataset manifest
- publish the dataset version to S3
- point Slurm training jobs at the dataset version

## How It Connects To The Existing Repo

The existing flow already has:

- training metrics
- eval results
- export manifests
- artifact publication
- commercial reporting

The data pipeline should produce one additional artifact:

- `dataset_manifest.json`

That manifest becomes the dataset-side counterpart to the training checkpoint manifest and the export manifest.

Recommended machine-readable linkage:

- `dataset_manifest.json`
  - `dataset_version`
  - `source_uri`
  - `raw_checksum`
  - `output_checksum`
  - `token_count`
  - `dedupe_method`
  - `build_id`
- `training_metrics.json`
  - `dataset_version`
  - `checkpoint_path`
  - `run_id`
- `export_manifest.json`
  - `source_checkpoint`
  - `source_step`
  - `artifact_checksum`
- eval result JSON
  - `checkpoint_path`
  - `perplexity`
  - `passed`

Then the full loop becomes:

```text
raw data -> dataset artifact -> training checkpoint -> eval -> published model -> commercial report
```

## What Success Looks Like

You have a credible pipeline when you can say:

- this dataset version came from these sources
- these filters were applied
- these duplicates were removed
- these tokens were packed this way
- this checkpoint was trained on that dataset version
- this eval result came from that checkpoint

That is the level of traceability and scale a serious buyer or lab reviewer expects.

## Implementation Checklist

See [docs/data-pipeline-implementation-checklist.md](data-pipeline-implementation-checklist.md) for the step-by-step build order, file list, and gates.
