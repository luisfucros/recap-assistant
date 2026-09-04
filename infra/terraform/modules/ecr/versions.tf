terraform {
  required_version = ">= 1.15.0, < 1.17.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.57.1, < 7.0.0"
    }
  }
}
