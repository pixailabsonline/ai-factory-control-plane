terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

# --- AMI: Ubuntu 22.04 with NVIDIA drivers ---

data "aws_ami" "ubuntu_gpu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# --- Networking ---

resource "aws_security_group" "training" {
  name        = "ai-factory-training"
  description = "AI Factory training instances — SSH + inter-node NCCL"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "NCCL rendezvous"
    from_port   = 29500
    to_port     = 29500
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "All traffic within training cluster"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project = "ai-factory-control-plane"
  }
}

# --- IAM: training instance role ---

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

  tags = {
    Project = "ai-factory-control-plane"
  }
}

resource "aws_iam_role_policy" "training_s3" {
  name = "ai-factory-s3-checkpoints"
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

resource "aws_iam_instance_profile" "training" {
  name = "ai-factory-training"
  role = aws_iam_role.training.name
}

# --- S3: checkpoint storage ---

resource "aws_s3_bucket" "checkpoints" {
  bucket = "ai-factory-checkpoints-${data.aws_caller_identity.current.account_id}"

  tags = {
    Project = "ai-factory-control-plane"
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

    filter {
      prefix = "checkpoints/"
    }
  }
}

# --- SSH Key ---

resource "aws_key_pair" "training" {
  key_name   = var.key_name
  public_key = file("~/.ssh/${var.key_name}.pub")

  tags = {
    Project = "ai-factory-control-plane"
  }
}

# --- Training instances ---

locals {
  instance_count = var.training_enabled ? (var.multi_node ? 2 : 1) : 0
}

resource "aws_instance" "training" {
  count = local.instance_count

  ami                    = data.aws_ami.ubuntu_gpu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.training.key_name
  vpc_security_group_ids = [aws_security_group.training.id]
  iam_instance_profile   = aws_iam_instance_profile.training.name
  user_data              = file("${path.module}/user_data.sh")

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
  }

  tags = {
    Name    = "ai-factory-training-${count.index}"
    Project = "ai-factory-control-plane"
    Role    = count.index == 0 ? "master" : "worker"
  }
}

# --- Placement group for multi-node (low-latency inter-node) ---

resource "aws_placement_group" "training" {
  count    = var.multi_node ? 1 : 0
  name     = "ai-factory-cluster"
  strategy = "cluster"

  tags = {
    Project = "ai-factory-control-plane"
  }
}
