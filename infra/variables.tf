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
  nullable    = false

  validation {
    condition     = length(var.allowed_ssh_cidrs) > 0
    error_message = "allowed_ssh_cidrs must be set explicitly (for example: [\"203.0.113.10/32\"])."
  }
}

variable "instance_type" {
  description = "GPU instance type for training"
  type        = string
  default     = "g5.xlarge"
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

variable "lustre_enabled" {
  description = "Create FSx Lustre shared filesystem for checkpoints and training data."
  type        = bool
  default     = false
}

variable "lustre_storage_gb" {
  description = "FSx Lustre storage capacity in GB (must be multiple of 1200)."
  type        = number
  default     = 1200
}
