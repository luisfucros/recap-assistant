# Recap AWS Terraform (Milestone 10)
#
# Application code is unchanged; this directory is the production deploy.
# Remote state is **not** created here — copy `backend.hcl.example` to
# `backend.hcl` and point it at the S3 bucket + DynamoDB lock table you
# already have.
#
# Config is Terraform variables → ECS task env later. No Secrets Manager,
# no Parameter Store.
#
# ## Layout
#
# - Root module wires what exists today: `vpc` + `ecr`.
# - Other folders under `modules/` are stubs (README only) for later slices
#   (RDS, Redis, Qdrant, ECS, S3 app buckets, CloudFront, monitoring, IAM).
#
# ## Prerequisites
#
# - Terraform **1.15.x** (not 1.16 yet — 1.16.0 is release-week). `tfenv` /
#   `asdf` will pick up `.terraform-version`.
# - AWS credentials that can manage VPC + ECR in `var.aws_region`.
# - Docker + AWS CLI (to build/push images after ECR exists).
# - An existing state backend.
#
# ## First apply
#
# ECR must exist before images can be pushed, and images must exist before
# later ECS tasks (and Qdrant/Prometheus/Grafana) can start. The Makefile
# encodes that order:
#
# ```bash
# cp backend.hcl.example backend.hcl          # edit bucket / key / table
# cp environments/dev.tfvars.example environments/dev.tfvars
# make tf-init
# make aws-bootstrap                          # ECR → linux/amd64 images → rest
# ```
#
# Equivalent step-by-step (each `apply` asks for confirmation unless
# `TF_AUTO_APPROVE=1`):
#
# ```bash
# make tf-apply-ecr                           # repos + ECR Public pull-through
# make images-push                            # git SHA + latest → recap-{api,ingestion,migrate}
# make tf-apply                               # VPC today; ECS/RDS in later slices
# ```
#
# Images are built `--platform=linux/amd64` (Fargate). Override the tag with
# `IMAGE_TAG=...` if you are not pushing `git rev-parse HEAD`.
#
# `make tf-destroy` tears down whatever this stack manages. It does **not**
# touch the operator-owned state backend. Repos that still contain images
# will fail to delete until those images are expired or deleted.
#
# ## Versions
#
# | Piece            | Pin      | Why |
# |------------------|----------|-----|
# | Terraform        | 1.15.x   | 1.16.0 shipped 26 Aug 2026; 1.15.9 is the prior minor's latest patch |
# | hashicorp/aws    | 6.57.1   | 6.62.0 is days old; 6.57.1 is the last patched minor (29 Jul 2026) |
