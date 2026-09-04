# IAM / extra security — not implemented yet.

Tier security groups already live in `modules/vpc` (they have to exist
before RDS/ECS). This module is reserved for task roles, the GitHub Actions
OIDC push role, and any later WAF/CloudTrail wiring — not a second copy of
the SGs.
