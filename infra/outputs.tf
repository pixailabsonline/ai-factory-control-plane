output "training_instance_ids" {
  description = "IDs of training instances"
  value       = aws_instance.training[*].id
}

output "training_public_ips" {
  description = "Public IPs of training instances"
  value       = aws_instance.training[*].public_ip
}

output "master_ip" {
  description = "Public IP of master node (for multi-node NCCL rendezvous)"
  value       = length(aws_instance.training) > 0 ? aws_instance.training[0].public_ip : null
}

output "checkpoint_bucket" {
  description = "S3 bucket for checkpoint storage"
  value       = aws_s3_bucket.checkpoints.bucket
}

output "security_group_id" {
  description = "Security group ID for training instances"
  value       = aws_security_group.training.id
}
