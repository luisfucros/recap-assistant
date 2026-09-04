output "vpc_id" {
  description = "VPC id for later ECS / RDS / ElastiCache modules."
  value       = module.vpc.vpc_id
}

output "public_subnet_ids" {
  description = "Public subnets (ALB, NAT)."
  value       = module.vpc.public_subnet_ids
}

output "private_app_subnet_ids" {
  description = "Private subnets for Fargate tasks."
  value       = module.vpc.private_app_subnet_ids
}

output "private_data_subnet_ids" {
  description = "Private subnets for RDS, ElastiCache, Qdrant."
  value       = module.vpc.private_data_subnet_ids
}

output "security_group_ids" {
  description = "Tier security groups keyed by role (alb, api, workers, data, monitoring)."
  value       = module.vpc.security_group_ids
}

output "aws_region" {
  description = "Region used for AWS CLI (ECR login)."
  value       = var.aws_region
}

output "ecr_registry_host" {
  description = "ECR registry host for `docker login` (account.dkr.ecr.region.amazonaws.com)."
  value       = split("/", module.ecr.repository_urls["api"])[0]
}

output "ecr_repository_urls" {
  description = "Push/pull URIs keyed by Dockerfile image (api, ingestion, migrate)."
  value       = module.ecr.repository_urls
}

output "ecr_repository_url_api" {
  description = "URI for the recap-api image (HTTP API, eval, eval-beat, checkpointer-setup)."
  value       = module.ecr.repository_urls["api"]
}

output "ecr_repository_url_ingestion" {
  description = "URI for the recap-ingestion image (worker + beat)."
  value       = module.ecr.repository_urls["ingestion"]
}

output "ecr_repository_url_migrate" {
  description = "URI for the recap-migrate image (alembic upgrade head)."
  value       = module.ecr.repository_urls["migrate"]
}

output "ecr_pull_through_prefixes" {
  description = "ECR repository prefixes for cached public images (no Docker Hub credentials, so ECR Public only)."
  value       = module.ecr.pull_through_prefixes
}
