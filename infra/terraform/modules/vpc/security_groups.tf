# Tier security groups. Ingress is cross-referenced by group id so we never
# open data stores to the VPC CIDR. Egress HTTPS from app/workers is required
# for hosted LLM / search / OAuth; tightening that further belongs with the
# ECS task definitions once destinations are known.

resource "aws_security_group" "alb" {
  name        = "${var.name}-alb"
  description = "Public ALB: HTTP/HTTPS in, nothing else."
  vpc_id      = aws_vpc.this.id

  tags = {
    Name = "${var.name}-alb"
    Tier = "alb"
  }
}

resource "aws_vpc_security_group_ingress_rule" "alb_http" {
  security_group_id = aws_security_group.alb.id
  description       = "HTTP (redirect to HTTPS once ACM is wired)"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 80
  to_port           = 80
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "alb_https" {
  security_group_id = aws_security_group.alb.id
  description       = "HTTPS"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "alb_all" {
  security_group_id = aws_security_group.alb.id
  description       = "ALB health checks and forwarded traffic to tasks"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_security_group" "api" {
  name        = "${var.name}-api"
  description = "HTTP API tasks: traffic from the ALB only."
  vpc_id      = aws_vpc.this.id

  tags = {
    Name = "${var.name}-api"
    Tier = "api"
  }
}

resource "aws_vpc_security_group_ingress_rule" "api_from_alb" {
  security_group_id            = aws_security_group.api.id
  description                  = "API HTTP from ALB"
  referenced_security_group_id = aws_security_group.alb.id
  from_port                    = 8000
  to_port                      = 8000
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "api_metrics_from_monitoring" {
  security_group_id            = aws_security_group.api.id
  description                  = "Prometheus scrape of /metrics"
  referenced_security_group_id = aws_security_group.monitoring.id
  from_port                    = 8000
  to_port                      = 8000
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "monitoring_scrape_api" {
  security_group_id            = aws_security_group.monitoring.id
  referenced_security_group_id = aws_security_group.api.id
  from_port                    = 8000
  to_port                      = 8000
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "api_https" {
  security_group_id = aws_security_group.api.id
  description       = "Hosted LLM, OAuth, web search, AWS APIs"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "api_http" {
  security_group_id = aws_security_group.api.id
  description       = "Internal HTTP (Qdrant, ALB health-check replies)"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 80
  to_port           = 80
  ip_protocol       = "tcp"
}

resource "aws_security_group" "workers" {
  name        = "${var.name}-workers"
  description = "Ingestion/eval workers and beats: no public ingress."
  vpc_id      = aws_vpc.this.id

  tags = {
    Name = "${var.name}-workers"
    Tier = "workers"
  }
}

resource "aws_vpc_security_group_egress_rule" "workers_https" {
  security_group_id = aws_security_group.workers.id
  description       = "Hosted embeddings and AWS APIs"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
}

resource "aws_security_group" "data" {
  name        = "${var.name}-data"
  description = "Postgres, Redis, Qdrant: ingress from api and workers only."
  vpc_id      = aws_vpc.this.id

  tags = {
    Name = "${var.name}-data"
    Tier = "data"
  }
}

resource "aws_vpc_security_group_ingress_rule" "data_postgres_api" {
  security_group_id            = aws_security_group.data.id
  description                  = "Postgres from API"
  referenced_security_group_id = aws_security_group.api.id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "data_postgres_workers" {
  security_group_id            = aws_security_group.data.id
  description                  = "Postgres from workers"
  referenced_security_group_id = aws_security_group.workers.id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "data_redis_api" {
  security_group_id            = aws_security_group.data.id
  description                  = "Redis from API"
  referenced_security_group_id = aws_security_group.api.id
  from_port                    = 6379
  to_port                      = 6379
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "data_redis_workers" {
  security_group_id            = aws_security_group.data.id
  description                  = "Redis from workers"
  referenced_security_group_id = aws_security_group.workers.id
  from_port                    = 6379
  to_port                      = 6379
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "data_qdrant_api" {
  security_group_id            = aws_security_group.data.id
  description                  = "Qdrant HTTP from API"
  referenced_security_group_id = aws_security_group.api.id
  from_port                    = 6333
  to_port                      = 6333
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "data_qdrant_workers" {
  security_group_id            = aws_security_group.data.id
  description                  = "Qdrant HTTP from workers"
  referenced_security_group_id = aws_security_group.workers.id
  from_port                    = 6333
  to_port                      = 6333
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "data_qdrant_grpc_api" {
  security_group_id            = aws_security_group.data.id
  description                  = "Qdrant gRPC from API"
  referenced_security_group_id = aws_security_group.api.id
  from_port                    = 6334
  to_port                      = 6334
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "data_qdrant_grpc_workers" {
  security_group_id            = aws_security_group.data.id
  description                  = "Qdrant gRPC from workers"
  referenced_security_group_id = aws_security_group.workers.id
  from_port                    = 6334
  to_port                      = 6334
  ip_protocol                  = "tcp"
}

resource "aws_security_group" "monitoring" {
  name        = "${var.name}-monitoring"
  description = "Prometheus/Grafana: no public listener; scraped internally later."
  vpc_id      = aws_vpc.this.id

  tags = {
    Name = "${var.name}-monitoring"
    Tier = "monitoring"
  }
}

resource "aws_vpc_security_group_ingress_rule" "monitoring_grafana" {
  security_group_id            = aws_security_group.monitoring.id
  description                  = "Grafana UI from the API SG (bastion/VPN later replaces this)"
  referenced_security_group_id = aws_security_group.api.id
  from_port                    = 3000
  to_port                      = 3000
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "monitoring_https" {
  security_group_id = aws_security_group.monitoring.id
  description       = "AWS APIs"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
}

# App and workers must reach data-plane ports (Postgres/Redis/Qdrant).
resource "aws_vpc_security_group_egress_rule" "api_to_data_postgres" {
  security_group_id            = aws_security_group.api.id
  referenced_security_group_id = aws_security_group.data.id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "api_to_data_redis" {
  security_group_id            = aws_security_group.api.id
  referenced_security_group_id = aws_security_group.data.id
  from_port                    = 6379
  to_port                      = 6379
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "api_to_data_qdrant" {
  security_group_id            = aws_security_group.api.id
  referenced_security_group_id = aws_security_group.data.id
  from_port                    = 6333
  to_port                      = 6334
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "workers_to_data_postgres" {
  security_group_id            = aws_security_group.workers.id
  referenced_security_group_id = aws_security_group.data.id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "workers_to_data_redis" {
  security_group_id            = aws_security_group.workers.id
  referenced_security_group_id = aws_security_group.data.id
  from_port                    = 6379
  to_port                      = 6379
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "workers_to_data_qdrant" {
  security_group_id            = aws_security_group.workers.id
  referenced_security_group_id = aws_security_group.data.id
  from_port                    = 6333
  to_port                      = 6334
  ip_protocol                  = "tcp"
}
