output "repository_urls" {
  description = "Repository URLs keyed by image (api, ingestion, migrate)."
  value       = { for key, repo in aws_ecr_repository.this : key => repo.repository_url }
}

output "repository_arns" {
  description = "Repository ARNs keyed by image."
  value       = { for key, repo in aws_ecr_repository.this : key => repo.arn }
}

output "registry_id" {
  description = "Account id of the ECR registry (same for every repo)."
  value       = aws_ecr_repository.this["api"].registry_id
}

output "pull_through_prefixes" {
  description = "Prefixes Fargate should use instead of Docker Hub."
  value = {
    ecr_public = aws_ecr_pull_through_cache_rule.ecr_public.ecr_repository_prefix
  }
}
