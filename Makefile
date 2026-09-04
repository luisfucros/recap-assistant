# Local Docker Compose and AWS Terraform entrypoints.
#
# AWS order matters once ECS exists: create ECR, push linux/amd64 images,
# then apply the rest (VPC today; later RDS/ECS pull those tags).

.DEFAULT_GOAL := help

COMPOSE      ?= docker compose
COMPOSE_TEST ?= docker compose -f docker-compose.test.yml
TF_DIR       ?= infra/terraform
BACKEND      ?= $(TF_DIR)/backend.hcl
TFVARS       ?= $(TF_DIR)/environments/dev.tfvars
TF           ?= terraform -chdir=$(TF_DIR)
TF_APPLY_FLAGS ?=

# Fargate is linux/amd64. Apple Silicon must set this or the image will not run.
DOCKER_PLATFORM ?= linux/amd64
IMAGE_TAG       ?= $(shell git rev-parse HEAD)

ifeq ($(TF_AUTO_APPROVE),1)
override TF_APPLY_FLAGS += -auto-approve
endif

.PHONY: help
help: ## List targets
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} /^[a-zA-Z0-9_.-]+:.*?##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: up
up: ## Build and start the local stack (foreground)
	$(COMPOSE) up --build

.PHONY: up-d
up-d: ## Build and start the local stack (detached)
	$(COMPOSE) up --build -d

.PHONY: watch
watch: ## Dev mode: compose watch (sync + reload)
	$(COMPOSE) watch

.PHONY: down
down: ## Stop the local stack (keep volumes)
	$(COMPOSE) down

.PHONY: down-v
down-v: ## Stop the local stack and delete named volumes
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Follow local stack logs
	$(COMPOSE) logs -f

.PHONY: ps
ps: ## Show local stack status
	$(COMPOSE) ps

.PHONY: test-infra
test-infra: ## Start throwaway Postgres/Qdrant/MinIO for pytest -m integration
	$(COMPOSE_TEST) up -d

.PHONY: test-infra-down
test-infra-down: ## Tear down throwaway test infra (including tmpfs)
	$(COMPOSE_TEST) down -v

.PHONY: test-unit
test-unit: ## Fast pytest unit suite (no I/O)
	uv run pytest -m unit

.PHONY: tf-init
tf-init: ## terraform init against the operator-provided backend.hcl
	@test -f $(BACKEND) || { echo "Missing $(BACKEND). Copy $(TF_DIR)/backend.hcl.example and fill in your existing bucket/table."; exit 1; }
	$(TF) init -backend-config=$(abspath $(BACKEND))

.PHONY: tf-fmt
tf-fmt: ## terraform fmt (recursive)
	$(TF) fmt -recursive

.PHONY: tf-validate
tf-validate: ## terraform validate (run tf-init first)
	$(TF) validate

.PHONY: _require-tfvars
_require-tfvars:
	@test -f $(TFVARS) || { echo "Missing $(TFVARS). Copy $(TF_DIR)/environments/dev.tfvars.example."; exit 1; }

.PHONY: tf-plan
tf-plan: _require-tfvars ## terraform plan (full stack)
	$(TF) plan -var-file=$(abspath $(TFVARS))

.PHONY: tf-plan-ecr
tf-plan-ecr: _require-tfvars ## terraform plan for ECR only
	$(TF) plan -var-file=$(abspath $(TFVARS)) -target=module.ecr

.PHONY: tf-apply-ecr
tf-apply-ecr: _require-tfvars ## Create/update ECR repos and pull-through cache (no VPC/ECS)
	$(TF) apply $(TF_APPLY_FLAGS) -var-file=$(abspath $(TFVARS)) -target=module.ecr

# Region comes from tfvars, not Terraform state — new outputs are missing until
# the next apply, and images-push must work right after the first tf-apply-ecr.
# Repo URLs come from ecr_repository_urls (present as soon as module.ecr exists).
tfvars_aws_region = $(shell awk -F'"' '/^[[:space:]]*aws_region[[:space:]]*=/{print $$2; exit}' $(TFVARS))

.PHONY: ecr-login
ecr-login: _require-tfvars ## docker login to this account's ECR (requires tf-apply-ecr)
	@region="$(tfvars_aws_region)"; \
	test -n "$$region" || { echo "Could not read aws_region from $(TFVARS)."; exit 1; }; \
	url="$$($(TF) output -json ecr_repository_urls | python3 -c 'import json,sys; print(json.load(sys.stdin)["api"])')"; \
	host="$${url%%/*}"; \
	test -n "$$host" || { echo "Could not read ECR URL from terraform state. Run make tf-apply-ecr first."; exit 1; }; \
	aws ecr get-login-password --region "$$region" | docker login --username AWS --password-stdin "$$host"

.PHONY: images-push
images-push: ecr-login ## Build linux/amd64 images and push git-SHA + latest tags
	$(MAKE) image-push IMAGE_KEY=api DOCKERFILE=services/api/Dockerfile
	$(MAKE) image-push IMAGE_KEY=ingestion DOCKERFILE=services/ingestion/Dockerfile
	$(MAKE) image-push IMAGE_KEY=migrate DOCKERFILE=docker/migrate.Dockerfile

.PHONY: image-push
image-push:
	@test -n "$(IMAGE_KEY)" && test -n "$(DOCKERFILE)"
	@url="$$($(TF) output -json ecr_repository_urls | python3 -c 'import json,sys; print(json.load(sys.stdin)["$(IMAGE_KEY)"])')"; \
	test -n "$$url" || { echo "Unknown IMAGE_KEY=$(IMAGE_KEY) in ecr_repository_urls."; exit 1; }; \
	echo "Building $(DOCKERFILE) → $$url:$(IMAGE_TAG) (platform $(DOCKER_PLATFORM))"; \
	docker build --platform=$(DOCKER_PLATFORM) -f $(DOCKERFILE) -t "$$url:$(IMAGE_TAG)" -t "$$url:latest" .; \
	docker push "$$url:$(IMAGE_TAG)"; \
	docker push "$$url:latest"

.PHONY: tf-apply
tf-apply: _require-tfvars ## Apply remaining Terraform (VPC today; later ECS after images-push)
	$(TF) apply $(TF_APPLY_FLAGS) -var-file=$(abspath $(TFVARS))

.PHONY: aws-bootstrap
aws-bootstrap: ## Sequential AWS deploy: ECR → push images → rest of the stack
	$(MAKE) tf-apply-ecr
	$(MAKE) images-push
	$(MAKE) tf-apply

.PHONY: tf-destroy
tf-destroy: _require-tfvars ## Destroy managed resources (leaves the remote state backend intact)
	$(TF) destroy $(TF_APPLY_FLAGS) -var-file=$(abspath $(TFVARS))
