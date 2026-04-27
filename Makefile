.PHONY: train profile scaling eval serve bench infra-init infra-plan infra-up infra-down cost-check

# --- Slurm jobs (run on the cluster) ---

train:
	sbatch slurm/train-single-node.sbatch

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
	cd infra && terraform plan

infra-up:
	cd infra && terraform apply -var="training_enabled=true"

infra-down:
	cd infra && terraform apply -var="training_enabled=false"

infra-multi:
	cd infra && terraform apply -var="training_enabled=true" -var="multi_node=true"

cost-check:
	bash infra/cost-check.sh

# --- Cluster status ---

jobs:
	squeue -u $$USER -o "%.8i %.20j %.4t %.10M %.6D %R"

gpu-status:
	sinfo -p gpu -N -o "%N %G %T %m %e"

logs:
	ls -lt logs/ | head -20
