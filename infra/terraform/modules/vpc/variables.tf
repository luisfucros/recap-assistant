variable "name" {
  description = "Name prefix applied to VPC resources (typically <project>-<environment>)."
  type        = string
}

variable "cidr_block" {
  description = "VPC IPv4 CIDR. Public / private-app / private-data /24s are carved from this."
  type        = string
}

variable "az_count" {
  description = "Number of AZs to use (minimum 2 for a later multi-AZ ALB)."
  type        = number
}

variable "enable_nat_gateway" {
  description = "When true, one zonal NAT Gateway in the first public subnet (cost-conscious default vs one NAT per AZ)."
  type        = bool
}

variable "aws_region" {
  description = "Region name for the S3 gateway endpoint service (com.amazonaws.<region>.s3)."
  type        = string
}
