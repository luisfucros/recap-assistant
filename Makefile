# Local Docker Compose and AWS Terraform entrypoints.

.DEFAULT_GOAL := help

COMPOSE      ?= docker compose
COMPOSE_TEST ?= docker compose -f docker-compose.test.yml
TF_DIR       ?= infra/terraform
BACKEND      ?= $(TF_DIR)/backend.hcl
TFVARS       ?= $(TF_DIR)/environments/dev.tfvars
TF           ?= terraform -chdir=$(TF_DIR)

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

.PHONY: tf-plan
tf-plan: ## terraform plan with environments/dev.tfvars
	@test -f $(TFVARS) || { echo "Missing $(TFVARS). Copy $(TF_DIR)/environments/dev.tfvars.example."; exit 1; }
	$(TF) plan -var-file=$(abspath $(TFVARS))

.PHONY: tf-apply
tf-apply: ## terraform apply with environments/dev.tfvars (interactive)
	@test -f $(TFVARS) || { echo "Missing $(TFVARS). Copy $(TF_DIR)/environments/dev.tfvars.example."; exit 1; }
	$(TF) apply -var-file=$(abspath $(TFVARS))

.PHONY: tf-destroy
tf-destroy: ## Destroy VPC + ECR (leaves the remote state backend intact)
	@test -f $(TFVARS) || { echo "Missing $(TFVARS). Copy $(TF_DIR)/environments/dev.tfvars.example."; exit 1; }
	$(TF) destroy -var-file=$(abspath $(TFVARS))
