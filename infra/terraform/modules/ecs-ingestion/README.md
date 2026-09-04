# ECS ingestion (worker + beat) — not implemented yet.

Both pull `module.ecr.repository_urls.ingestion`. Ingestion-beat is
`desired_count = 1` and must not share eval-beat.
