terraform {
  # 1.16.0 shipped 26 Aug 2026 (release week). 1.15 is the prior minor;
  # 1.15.9 is its latest patch (19 Aug 2026) and is still in security support.
  required_version = ">= 1.15.0, < 1.17.0"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # 6.62.0 is days old (26 Aug 2026). 6.57.1 is the last patched minor
      # before a run of weekly .0 cuts (released 29 Jul 2026).
      version = "6.57.1"
    }
  }
}
