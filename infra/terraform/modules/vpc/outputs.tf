output "vpc_id" {
  description = "VPC id."
  value       = aws_vpc.this.id
}

output "vpc_cidr_block" {
  description = "VPC CIDR (useful for later SG rules)."
  value       = aws_vpc.this.cidr_block
}

output "public_subnet_ids" {
  description = "Public subnet ids, one per AZ, stable AZ order."
  value       = [for az in local.azs : aws_subnet.public[az].id]
}

output "private_app_subnet_ids" {
  description = "Private application subnet ids, one per AZ, stable AZ order."
  value       = [for az in local.azs : aws_subnet.private_app[az].id]
}

output "private_data_subnet_ids" {
  description = "Private data subnet ids, one per AZ, stable AZ order."
  value       = [for az in local.azs : aws_subnet.private_data[az].id]
}

output "nat_gateway_id" {
  description = "Zonal NAT Gateway id, or null when enable_nat_gateway is false."
  value       = var.enable_nat_gateway ? aws_nat_gateway.this[0].id : null
}

output "security_group_ids" {
  description = "Tier security group ids."
  value = {
    alb         = aws_security_group.alb.id
    api         = aws_security_group.api.id
    workers     = aws_security_group.workers.id
    data        = aws_security_group.data.id
    monitoring  = aws_security_group.monitoring.id
  }
}
