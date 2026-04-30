.PHONY: train train-smoke train-recovery profile scaling eval export-model publish-model serve bench infra-init infra-plan infra-up infra-down cost-check jobs gpu-status substrate-status slurm-status platform-status logs

S3_BUCKET ?= ai-factory-checkpoints-737213639346
MODEL_NAME ?= gpt2-medium
S3_RUN_KEY ?= runs/$(MODEL_NAME)
S3_RUN_ROOT ?= s3://$(S3_BUCKET)/$(S3_RUN_KEY)

SSH_CIDR ?= 127.0.0.1/32

# --- Slurm jobs (run on the cluster) ---

train:
	sbatch --export=ALL,MAX_STEPS=$(MAX_STEPS),CKPT_EVERY=$(CKPT_EVERY) slurm/train-single-node.sbatch

train-smoke:
	sbatch slurm/train-smoke.sbatch

train-recovery:
	sbatch slurm/train-recovery.sbatch

train-multi:
	sbatch --export=ALL,MAX_STEPS=$(MAX_STEPS),CKPT_EVERY=$(CKPT_EVERY) slurm/train-multi-node.sbatch

profile:
	sbatch slurm/profile.sbatch

eval:
	sbatch --export=ALL,S3_BUCKET=$(S3_BUCKET),MODEL_NAME=$(MODEL_NAME),S3_RUN_KEY=$(S3_RUN_KEY),S3_RUN_ROOT=$(S3_RUN_ROOT) slurm/eval.sbatch

export-model:
	@test -n "$(CHECKPOINT)" || (echo "CHECKPOINT is required" && exit 1)
	@test -n "$(MODEL_DIR)" || (echo "MODEL_DIR is required" && exit 1)
	source /opt/training-env/bin/activate && python training/export_model.py \
		--checkpoint "$(CHECKPOINT)" \
		--output-dir "$(MODEL_DIR)" \
		$(if $(MODEL),--model "$(MODEL)",) \
		$(if $(TOKENIZER),--tokenizer "$(TOKENIZER)",)

publish-model:
	@test -n "$(CHECKPOINT)" || (echo "CHECKPOINT is required" && exit 1)
	@test -n "$(MODEL_DIR)" || (echo "MODEL_DIR is required" && exit 1)
	@test -n "$(MODEL_S3_URI)$(MODEL_S3_ROOT)" || (echo "MODEL_S3_URI or MODEL_S3_ROOT is required" && exit 1)
	@if [ -n "$(MODEL_S3_URI)" ]; then S3_FLAG="--s3-uri $(MODEL_S3_URI)"; else S3_FLAG="--s3-root $(MODEL_S3_ROOT)"; fi; \
	source /opt/training-env/bin/activate && python training/publish_model.py \
		--checkpoint "$(CHECKPOINT)" \
		--output-dir "$(MODEL_DIR)" \
		$$S3_FLAG \
		$(if $(MODEL),--model "$(MODEL)",) \
		$(if $(TOKENIZER),--tokenizer "$(TOKENIZER)",)

serve:
	@opts="--export=ALL,MODEL=$(MODEL),MODEL_DIR=$(MODEL_DIR),MODEL_S3_URI=$(MODEL_S3_URI),MODEL_S3_ROOT=$(MODEL_S3_ROOT),CHECKPOINT=$(CHECKPOINT),TOKENIZER=$(TOKENIZER),EXPORT_DIR=$(EXPORT_DIR),PORT=$(PORT),TP=$(TP)"; \
	if [ -n "$(NODELIST)" ]; then opts="$$opts --nodelist=$(NODELIST)"; fi; \
	sbatch $$opts slurm/serve.sbatch

scaling:
	source /opt/training-env/bin/activate && python training/scaling_bench.py \
		--model mistralai/Mistral-7B-v0.1 \
		--max-steps 100

bench:
	source /opt/training-env/bin/activate && python inference/bench.py \
		--url http://localhost:8080 --requests 50

# --- Infrastructure (Terraform) ---

infra-init:
	cd infra && terraform init

infra-plan:
	cd infra && terraform plan -var='allowed_ssh_cidrs=["$(SSH_CIDR)"]'

infra-up:
	cd infra && terraform apply -var="training_enabled=true" -var='allowed_ssh_cidrs=["$(SSH_CIDR)"]'

infra-down:
	cd infra && terraform apply -var="training_enabled=false" -var='allowed_ssh_cidrs=["$(SSH_CIDR)"]'

infra-multi:
	cd infra && terraform apply -var="training_enabled=true" -var="multi_node=true" -var='allowed_ssh_cidrs=["$(SSH_CIDR)"]'

cost-check:
	bash infra/cost-check.sh

# --- Cluster status ---

jobs:
	squeue -u $$USER -o "%.8i %.20j %.4t %.10M %.6D %R"

gpu-status:
	sinfo -p gpu -N -o "%N %G %T %m %e"

substrate-status:
	kubectl get nodes -L ai-factory/capacity-owner,ai-factory/slurm-pool,ai-factory/scheduler -o wide
	kubectl get pods -n gpu-operator -o wide

slurm-status:
	sinfo -Nel
	squeue -o "%.8i %.20j %.9P %.4t %.10M %.6D %R"
	scontrol show partition gpu
	scontrol show partition slurm-batch

platform-status: substrate-status slurm-status

logs:
	ls -lt logs/ | head -20
