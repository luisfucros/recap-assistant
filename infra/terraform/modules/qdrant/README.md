# Qdrant on ECS Fargate — not implemented yet.

Next M10 slices (compute). Single task, EBS volume, internal target group.
Admin API key is a Terraform-supplied env var, not Secrets Manager.

Place the task in **private-app** subnets (they have NAT for image pulls and
any outbound HTTPS). Data subnets have no default route on purpose.
