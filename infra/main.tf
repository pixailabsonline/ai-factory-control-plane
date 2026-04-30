terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "ai-factory-tfstate-af-ctrl-x7k2"
    key            = "infra/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "ai-factory-tf-locks-af-ctrl-x7k2"
    encrypt        = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "ai-factory-control-plane"
      ManagedBy   = "terraform"
      Environment = "training"
    }
  }
}

data "aws_caller_identity" "current" {}

# --- Remote state bucket (bootstrap manually - see backend block above) ---

# --- AMI: Deep Learning Base AMI (includes NVIDIA drivers + CUDA) ---

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["898082745236"] # AWS Deep Learning AMIs

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# --- Networking ---

resource "aws_security_group" "training" {
  name        = "ai-factory-training"
  description = "AI Factory training instances - SSH + inter-node NCCL"

  ingress {
    description = "SSH - operator only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  ingress {
    description = "NCCL rendezvous (torchrun)"
    from_port   = 29500
    to_port     = 29500
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "NCCL data plane - inter-node GPU communication"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  egress {
    description = "Outbound - pip, apt, S3, HuggingFace model downloads"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- IAM: training instance role (least privilege) ---

resource "aws_iam_role" "training" {
  name = "ai-factory-training"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "training_s3" {
  name = "s3-checkpoints"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.checkpoints.arn,
          "${aws_s3_bucket.checkpoints.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "training_cloudwatch" {
  name = "cloudwatch-logs"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          "arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:/ai-factory/*",
          "arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:/ai-factory/*:*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "AIFactory"
          }
        }
      }
    ]
  })
}

resource "aws_iam_instance_profile" "training" {
  name = "ai-factory-training"
  role = aws_iam_role.training.name
}

# --- S3: checkpoint storage ---

resource "aws_s3_bucket" "checkpoints" {
  bucket = "ai-factory-checkpoints-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    id     = "expire-old-checkpoints"
    status = "Enabled"

    expiration {
      days = 30
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }

    filter {
      prefix = "checkpoints/"
    }
  }

  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }

    filter {}
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- CloudWatch: log group for training logs ---

resource "aws_cloudwatch_log_group" "training" {
  name              = "/ai-factory/training"
  retention_in_days = 14
}

# --- CloudWatch: GPU utilization alarm ---

resource "aws_cloudwatch_metric_alarm" "gpu_idle" {
  count = local.instance_count

  alarm_name          = "ai-factory-gpu-idle-${count.index}"
  alarm_description   = "GPU utilization below 5% for 15 min - wasting credits"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 3
  metric_name         = "GPUUtilization"
  namespace           = "AIFactory"
  period              = 300
  statistic           = "Average"
  threshold           = 5
  treat_missing_data  = "missing"

  dimensions = {
    InstanceId = aws_instance.training[count.index].id
  }

  alarm_actions = var.alert_sns_arn != "" ? [var.alert_sns_arn] : []
}

# --- SSH Key ---

resource "aws_key_pair" "training" {
  key_name   = var.key_name
  public_key = file("~/.ssh/${var.key_name}.pub")
}

# --- Subnet: pick an AZ where the GPU instance type is available ---

data "aws_ec2_instance_type_offerings" "gpu" {
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
  location_type = "availability-zone"
}

data "aws_subnets" "gpu_capable" {
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
  filter {
    name   = "availability-zone"
    values = [data.aws_ec2_instance_type_offerings.gpu.locations[0]]
  }
}

# --- Placement group for multi-node (co-locate for low-latency NCCL) ---

resource "aws_placement_group" "training" {
  count    = var.multi_node ? 1 : 0
  name     = "ai-factory-cluster"
  strategy = "cluster"
}

# --- Training instances ---

locals {
  instance_count = var.training_enabled ? (var.multi_node ? 2 : 1) : 0
  hourly_prices = {
    "p3.2xlarge"    = 3.06
    "p3.8xlarge"    = 12.24
    "p3.16xlarge"   = 24.48
    "p4d.24xlarge"  = 32.77
    "p4de.24xlarge" = 40.96
    "p5.48xlarge"   = 98.32
    "g5.xlarge"     = 1.006
    "g5.2xlarge"    = 1.212
    "g5.12xlarge"   = 5.672
    "g4dn.xlarge"   = 0.526
    "g6.xlarge"     = 0.805
    "g6.12xlarge"   = 6.196
  }
  hourly_cost = lookup(local.hourly_prices, var.instance_type, null)
}

resource "aws_instance" "training" {
  count = local.instance_count

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.training.key_name
  subnet_id              = data.aws_subnets.gpu_capable.ids[0]
  vpc_security_group_ids = [aws_security_group.training.id]
  iam_instance_profile   = aws_iam_instance_profile.training.name
  user_data              = file("${path.module}/user_data.sh")
  placement_group        = var.multi_node ? aws_placement_group.training[0].id : null

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
  }

  metadata_options {
    http_tokens            = "required" # IMDSv2 only
    instance_metadata_tags = "enabled"  # allows user_data to read Role tag
  }

  ebs_optimized = true
  monitoring    = true

  tags = {
    Name  = "ai-factory-training-${count.index}"
    Role  = count.index == 0 ? "master" : "worker"
    Phase = var.multi_node ? "2-multi-node" : "1-single-node"
  }
}
