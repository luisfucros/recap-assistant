# One private repository per image this repo builds. ECS services that share a
# Dockerfile (api + eval + eval-beat + checkpointer-setup) share recap-api;
# they do not get their own repos.
#
# Tag mutability is immutable except for `latest`, which CI overwrites as a
# convenience alias. Task definitions must still pin the git SHA.

locals {
  repositories = {
    api = {
      name        = "${var.name_prefix}-api"
      dockerfile  = "services/api/Dockerfile"
      description = "HTTP API eval worker eval-beat checkpointer-setup"
    }
    ingestion = {
      name        = "${var.name_prefix}-ingestion"
      dockerfile  = "services/ingestion/Dockerfile"
      description = "Ingestion worker and ingestion-beat"
    }
    migrate = {
      name        = "${var.name_prefix}-migrate"
      dockerfile  = "docker/migrate.Dockerfile"
      description = "One-shot alembic upgrade head"
    }
  }
}

resource "aws_ecr_repository" "this" {
  for_each = local.repositories

  name                 = each.value.name
  image_tag_mutability = "IMMUTABLE_WITH_EXCLUSION"

  image_tag_mutability_exclusion_filter {
    filter      = "latest"
    filter_type = "WILDCARD"
  }

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name       = each.value.name
    Dockerfile = each.value.dockerfile
    UsedBy     = each.value.description
  }
}

resource "aws_ecr_lifecycle_policy" "this" {
  for_each = aws_ecr_repository.this

  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after ${var.untagged_image_expiry_days} days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = var.untagged_image_expiry_days
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Keep last ${var.tagged_image_keep_count} tagged images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = var.tagged_image_keep_count
        }
        action = { type = "expire" }
      }
    ]
  })
}

# Docker Hub pull-through needs a Secrets Manager credential; this learning
# deploy does not use SM. ECR Public does not, and covers Qdrant / Prometheus
# / Grafana well enough for Fargate. Tasks pull
# <account>.dkr.ecr.<region>.amazonaws.com/ecr-public/<upstream-path>.
resource "aws_ecr_pull_through_cache_rule" "ecr_public" {
  ecr_repository_prefix = "ecr-public"
  upstream_registry_url = "public.ecr.aws"
}
