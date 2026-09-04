# VPC, subnets, a single NAT Gateway, an S3 gateway endpoint, and per-tier
# security groups for the Recap AWS stack.
#
# Why a single NAT: one NAT per AZ is the HA default and roughly 2–3× the
# monthly cost. This is a learning deploy; `enable_nat_gateway = false` is
# the escape hatch if you only need AWS APIs (S3 already has a gateway
# endpoint). Hosted LLM calls still need NAT (or later, HTTPS proxy).
# Data subnets have no default route — RDS/Redis/Qdrant cannot initiate
# outbound internet.
#
# ## Usage
#
# ```hcl
# module "vpc" {
#   source             = "./modules/vpc"
#   name               = "recap-dev"
#   aws_region         = "us-east-1"
#   cidr_block         = "10.42.0.0/16"
#   az_count           = 2
#   enable_nat_gateway = true
# }
# ```
