# Postgres (RDS Aurora) — not implemented yet.

Next M10 slice. Will consume `module.vpc.private_data_subnet_ids` and
`module.vpc.security_group_ids.data`. Must **not** set
`manage_master_user_password` (that creates a billed Secrets Manager secret);
the master password is a Terraform variable injected as ECS env later.
