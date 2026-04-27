variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "key_name" {
  description = "SSH key pair name for training instances"
  type        = string
  default     = "ai-factory"
}

variable "instance_type" {
  description = "GPU instance type for training"
  type        = string
  default     = "p3.8xlarge"
}

variable "volume_size" {
  description = "Root volume size in GB (needs space for model weights, checkpoints, datasets)"
  type        = number
  default     = 200
}

variable "training_enabled" {
  description = "Set to true to launch the training instance. False = instance terminated, no cost."
  type        = bool
  default     = false
}

variable "multi_node" {
  description = "Set to true for Phase 2 multi-node (launches 2 instances)"
  type        = bool
  default     = false
}
