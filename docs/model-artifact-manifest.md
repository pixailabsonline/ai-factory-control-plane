# Model Artifact Manifest

Use this file type to record a trained model artifact without committing the binary itself.

## Why This Exists

- The checkpoint or exported model file is too large for git.
- The repo still needs evidence that the artifact exists, where it lives, and how it was validated.
- This lets you prove the trained output without storing the raw weights in the repository.

## What To Record

- Job ID
- Training run name
- Base model
- Final step
- Checkpoint path
- Artifact format
- Artifact size
- Artifact checksum
- Eval job ID
- Eval result
- Promotion decision
- Storage location

## Example

```yaml
run_name: stage3-multi-node-gpt2-medium
job_id: 25
base_model: gpt2-medium
final_step: 1000
checkpoint_path: /tmp/checkpoints/25/checkpoint-1000
artifact_format: checkpoint-directory
artifact_size_bytes: 123456789
artifact_checksum: sha256:...
eval_job_id: 26
eval_result: pass
perplexity: 78.48
threshold: 90.0
promotion_decision: promoted
storage_location: s3://<bucket>/models/...
notes: Trained on 2x A10G with FSDP FULL_SHARD.
```

## Suggested Rule

- For every training run you want to cite as evidence, create one manifest alongside the ops journal entry.
- Keep the manifest human-readable.
- Keep the raw model artifact in S3 or another artifact store.
- Prefer a run root like `s3://<bucket>/runs/<run-name>/` with checkpoints under `checkpoints/` and promoted models under `models/latest`.
- Reference the manifest from the journal entry.
