"""Pipeline configuration."""

# Separate models for SQL generation vs answer generation
DEFAULT_SQL_MODEL = "gpt-4o-mini"
DEFAULT_ANSWER_MODEL = "gpt-4o-mini"  # Can use gpt-4o for richer answers

# Fallback model if primary fails (API error, rate limit, etc.) — Anthropic
FALLBACK_SQL_MODEL = "claude-sonnet-4-20250514"
FALLBACK_ANSWER_MODEL = "claude-sonnet-4-20250514"
MAX_ROWS_FOR_ANSWER = 500
MAX_SQL_RETRIES = 3
