# Data Pipeline Justification

For this project, the data pipeline is the highest-leverage next step because it moves the story from "I can run training jobs" to "I can produce better training data and therefore better models more reliably."

Commercial buyers and frontier lab reviewers care about whether the pipeline can take messy raw data and turn it into a training-ready corpus that is:

- traceable
- clean
- deduplicated
- reproducible
- versioned
- free of benchmark contamination

## What matters

1. **Source provenance**
- Where did the data come from?
- Which version was used?
- What license or policy allowed it?

2. **Quality filtering**
- Remove spam, boilerplate, duplicates, and low-signal text.
- Show that bad data is filtered out before training.

3. **Deduplication**
- Remove exact and near-duplicate content.
- Prevent wasted tokens and poor generalization.
- Exact dedup is the first implementation target; near-dedup can come later.

4. **Tokenization and packing**
- Turn raw text into training-ready sequences efficiently.
- This directly affects throughput and cost.

5. **Dataset versioning**
- Every run should point to an immutable dataset version.
- You should be able to trace a model back to the exact data it used.

6. **Reproducibility**
- Same raw inputs plus same pipeline should produce the same dataset output.
- This makes the pipeline trustworthy.

7. **Contamination control**
- Prevent benchmark leakage.
- Keep train, validation, and evaluation data separate.
- Avoid accidental inclusion of eval data in training.

## Why this matters commercially

This is what a serious buyer wants to know:

- Can you reliably turn messy raw data into training-ready data?
- Can you prove what went into each model?
- Can you improve model quality with less wasted compute?

That is why data pipeline work is a stronger next step than more model-research claims right now. It is immediately legible to anyone who has run a real training job, and it is a direct lever on cost and model quality.

## What is missing today

The current repo proves:

- training works
- checkpointing works
- recovery works
- eval works
- export and publish work

What it does not yet prove is:

- a real data ingestion path
- cleaning and deduplication at corpus scale
- a versioned dataset artifact
- evidence that the data pipeline improved the model or reduced training cost

## Best next move

Build a small but real data pipeline that:

- ingests Common Crawl
- filters it
- exact-deduplicates it first, then add near-dedup later if needed
- tokenizes and packs it
- versions it in S3 with a manifest
- trains on it
- compares against the toy baseline
- reports cost per clean token and downstream eval delta

Run it at **1 TB of input** so the system is forced to prove operational stability, not just correctness.

That is the shortest path to a credible commercial data story.
