.PHONY: train train-smoke profile scaling eval serve bench infra-init infra-plan infra-up infra-down cost-check jobs gpu-status substrate-status slurm-status platform-status logs

SSH_CIDR ?= 127.0.0.1/32

# --- Slurm jobs (run on the cluster) ---

train:
	sbatch --export=ALL,MAX_STEPS=$(MAX_STEPS),CKPT_EVERY=$(CKPT_EVERY) slurm/train-single-node.sbatch

train-smoke:
	sbatch slurm/train-smoke.sbatch

train-multi:
	sbatch slurm/train-multi-node.sbatch

profile:
	sbatch slurm/profile.sbatch

eval:
	sbatch slurm/eval.sbatch $(CHECKPOINT)

serve:
	sbatch slurm/serve.sbatch

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
