locals {
  name = "${var.project}-${var.environment}"
}

module "vpc" {
  source = "./modules/vpc"

  name               = local.name
  aws_region         = var.aws_region
  cidr_block         = var.vpc_cidr
  az_count           = var.az_count
  enable_nat_gateway = var.enable_nat_gateway
}

module "ecr" {
  source = "./modules/ecr"

  name_prefix                 = var.project
  untagged_image_expiry_days  = var.untagged_image_expiry_days
  tagged_image_keep_count     = var.tagged_image_keep_count
}
