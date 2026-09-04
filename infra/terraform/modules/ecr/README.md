# Private ECR repositories for Recap images, plus an ECR Public pull-through
# cache so Fargate never hits Docker Hub rate limits.
#
# Repositories follow Dockerfiles, not Compose service names:
#
# | Repository        | Dockerfile                    | Pulled by                                      |
# |-------------------|-------------------------------|------------------------------------------------|
# | `<prefix>-api`    | `services/api/Dockerfile`     | api, eval, eval-beat, checkpointer-setup       |
# | `<prefix>-ingestion` | `services/ingestion/Dockerfile` | ingestion worker, ingestion-beat            |
# | `<prefix>-migrate` | `docker/migrate.Dockerfile`  | one-shot `alembic upgrade head`                |
#
# Image tags are immutable except `latest` (CI convenience alias). Task
# definitions should pin the git SHA.
#
# Docker Hub pull-through is intentionally omitted: AWS requires a Secrets
# Manager credential for it, and this stack does not use Secrets Manager.
#
# ## Usage
#
# ```hcl
# module "ecr" {
#   source                      = "./modules/ecr"
#   name_prefix                 = "recap"
#   untagged_image_expiry_days  = 7
#   tagged_image_keep_count     = 20
# }
# ```
