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

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH into training instances"
  type        = list(string)
  default     = ["0.0.0.0/0"] # Override in tfvars with your IP
}

variable "instance_type" {
  description = "GPU instance type for training"
  type        = string
  default     = "p3.8xlarge"
}

variable "volume_size" {
  description = "Root volume size in GB (model weights + checkpoints + datasets)"
  type        = number
  default     = 200
}

variable "training_enabled" {
  description = "Set to true to launch training instances. False = terminated, no cost."
  type        = bool
  default     = false
}

variable "multi_node" {
  description = "Set to true for Phase 2 multi-node (launches 2 instances in placement group)"
  type        = bool
  default     = false
}

variable "alert_sns_arn" {
  description = "SNS topic ARN for GPU idle alerts. Empty = no alerts."
  type        = string
  default     = ""
}
