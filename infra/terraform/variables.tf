variable "aws_region" {
  description = "AWS region for every resource in this stack."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Short environment name (dev, staging, prod). Used in resource names and default tags."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "project" {
  description = "Name prefix for AWS resources (ECR repos become <project>-api, etc.)."
  type        = string
  default     = "recap"
}

variable "vpc_cidr" {
  description = "IPv4 CIDR for the VPC. Must be large enough for public + private-app + private-data /24s across two AZs."
  type        = string
  default     = "10.42.0.0/16"
}

variable "az_count" {
  description = "How many AZs to span. Two is the ALB/RDS minimum without paying for a third NAT later."
  type        = number
  default     = 2

  validation {
    condition     = var.az_count >= 2 && var.az_count <= 3
    error_message = "az_count must be 2 or 3."
  }
}

variable "enable_nat_gateway" {
  description = "Create a single NAT Gateway so private tasks can reach hosted LLM/search APIs. The main ongoing cost of this slice; set false only if you accept no outbound internet from private subnets."
  type        = bool
  default     = true
}

variable "untagged_image_expiry_days" {
  description = "ECR lifecycle: expire untagged images after this many days."
  type        = number
  default     = 7
}

variable "tagged_image_keep_count" {
  description = "ECR lifecycle: keep this many tagged images per repo (git-SHA tags plus the moving latest alias)."
  type        = number
  default     = 20
}
