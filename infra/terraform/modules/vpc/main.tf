# Two AZs, three subnet tiers: public (ALB + NAT), private-app (Fargate),
# private-data (RDS / Redis / Qdrant). A single NAT keeps the learning-deploy
# bill down; private tasks share it for hosted-API egress.

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, var.az_count)

  # /16 → three /20-ish bands of /24s: .0 public, .16 app, .32 data.
  public_cidrs       = [for i, az in local.azs : cidrsubnet(var.cidr_block, 8, i)]
  private_app_cidrs  = [for i, az in local.azs : cidrsubnet(var.cidr_block, 8, i + 16)]
  private_data_cidrs = [for i, az in local.azs : cidrsubnet(var.cidr_block, 8, i + 32)]
}

resource "aws_vpc" "this" {
  cidr_block           = var.cidr_block
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = var.name
  }
}

# The default SG allows all traffic inside the VPC; lock it so only the
# explicit tier groups we define below are usable.
resource "aws_default_security_group" "lockdown" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.name}-default-locked"
  }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.name}-igw"
  }
}

resource "aws_subnet" "public" {
  for_each = { for idx, az in local.azs : az => {
    index = idx
    cidr  = local.public_cidrs[idx]
  } }

  vpc_id                  = aws_vpc.this.id
  availability_zone       = each.key
  cidr_block              = each.value.cidr
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.name}-public-${each.key}"
    Tier = "public"
  }
}

resource "aws_subnet" "private_app" {
  for_each = { for idx, az in local.azs : az => {
    index = idx
    cidr  = local.private_app_cidrs[idx]
  } }

  vpc_id            = aws_vpc.this.id
  availability_zone = each.key
  cidr_block        = each.value.cidr

  tags = {
    Name = "${var.name}-app-${each.key}"
    Tier = "private-app"
  }
}

resource "aws_subnet" "private_data" {
  for_each = { for idx, az in local.azs : az => {
    index = idx
    cidr  = local.private_data_cidrs[idx]
  } }

  vpc_id            = aws_vpc.this.id
  availability_zone = each.key
  cidr_block        = each.value.cidr

  tags = {
    Name = "${var.name}-data-${each.key}"
    Tier = "private-data"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.name}-public"
  }
}

resource "aws_route" "public_internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.this.id
}

resource "aws_route_table_association" "public" {
  for_each = aws_subnet.public

  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

resource "aws_eip" "nat" {
  count  = var.enable_nat_gateway ? 1 : 0
  domain = "vpc"

  tags = {
    Name = "${var.name}-nat"
  }

  depends_on = [aws_internet_gateway.this]
}

resource "aws_nat_gateway" "this" {
  count = var.enable_nat_gateway ? 1 : 0

  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[local.azs[0]].id

  tags = {
    Name = "${var.name}-nat"
  }

  depends_on = [aws_internet_gateway.this]
}

resource "aws_route_table" "private_app" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.name}-private-app"
  }
}

resource "aws_route" "private_app_nat" {
  count = var.enable_nat_gateway ? 1 : 0

  route_table_id         = aws_route_table.private_app.id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.this[0].id
}

resource "aws_route_table" "private_data" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.name}-private-data"
  }
}

resource "aws_route_table_association" "private_app" {
  for_each = aws_subnet.private_app

  subnet_id      = each.value.id
  route_table_id = aws_route_table.private_app.id
}

resource "aws_route_table_association" "private_data" {
  for_each = aws_subnet.private_data

  subnet_id      = each.value.id
  route_table_id = aws_route_table.private_data.id
}

# Free: private S3 access without hairpinning through NAT (ECR layers also
# live on S3, so image pulls get cheaper once Fargate lands).
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.this.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids = [
    aws_route_table.private_app.id,
    aws_route_table.private_data.id,
    aws_route_table.public.id,
  ]

  tags = {
    Name = "${var.name}-s3"
  }
}
