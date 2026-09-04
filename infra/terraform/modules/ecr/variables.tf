variable "name_prefix" {
  description = "Repository name prefix. Repos are <prefix>-api, <prefix>-ingestion, <prefix>-migrate — one per Dockerfile, not per ECS service."
  type        = string
}

variable "untagged_image_expiry_days" {
  description = "Expire untagged images after this many days."
  type        = number
}

variable "tagged_image_keep_count" {
  description = "Keep at most this many tagged images (git SHA tags plus the moving latest alias)."
  type        = number
}
