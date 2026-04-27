.PHONY: build test infra-init infra-plan infra-up infra-down cost-check train profile scaling eval serve bench

# Go control plane
build:
	go build ./...

test:
	go test ./...

# Infrastructure (Terraform)
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

# Training (run on GPU instance)
train:
	torchrun --nproc_per_node=4 training/fsdp_trainer.py \
		--model mistralai/Mistral-7B-v0.1 \
		--batch-size 2 \
		--gradient-accumulation 8 \
		--max-steps 5000 \
		--checkpoint-every 500

# Profiling (run on GPU instance)
profile:
	torchrun --nproc_per_node=4 training/fsdp_trainer.py \
		--model mistralai/Mistral-7B-v0.1 \
		--batch-size 2 \
		--max-steps 50

# Scaling benchmark (run on GPU instance)
scaling:
	python training/scaling_bench.py \
		--model mistralai/Mistral-7B-v0.1 \
		--max-steps 100

# Eval (run on GPU instance)
eval:
	python eval/quality_gate.py \
		--checkpoint ./checkpoints/checkpoint-5000 \
		--max-perplexity 20.0

# Inference (run on GPU instance)
serve:
	python inference/server.py --model mistralai/Mistral-7B-v0.1 --checkpoint ./checkpoints/checkpoint-5000

bench:
	python inference/bench.py --url http://localhost:8080 --requests 50
