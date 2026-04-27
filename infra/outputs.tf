output "training_instance_ids" {
  description = "IDs of training instances"
  value       = aws_instance.training[*].id
}

output "training_public_ips" {
  description = "Public IPs of training instances (SSH targets)"
  value       = aws_instance.training[*].public_ip
}

output "master_ip" {
  description = "Master node IP (NCCL rendezvous endpoint for multi-node)"
  value       = length(aws_instance.training) > 0 ? aws_instance.training[0].public_ip : null
}

output "checkpoint_bucket" {
  description = "S3 bucket for checkpoint storage"
  value       = aws_s3_bucket.checkpoints.bucket
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for training logs"
  value       = aws_cloudwatch_log_group.training.name
}

output "ssh_command" {
  description = "SSH command to connect to master node"
  value       = length(aws_instance.training) > 0 ? "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_instance.training[0].public_ip}" : null
}

output "estimated_hourly_cost" {
  description = "Estimated hourly cost of running instances"
  value       = local.instance_count > 0 ? "${local.instance_count} x ${var.instance_type} ≈ $${local.instance_count * 12.24}/hr" : "No instances running"
}
