# Partial backend: bucket / key / region / lock table are supplied at
# `terraform init -backend-config=backend.hcl`. This stack never creates
# those resources — the operator already owns them.
terraform {
  backend "s3" {}
}
