# ECS API (HTTP + eval + eval-beat + checkpointer-setup) — not implemented yet.

All of these pull `module.ecr.repository_urls.api`. They are distinct task
definitions (command, CPU/memory, desired count), not distinct ECR repos.
One-shot migrate uses `repository_urls.migrate` instead.
