#!/usr/bin/env bash
# Fetch raw GitHub activity data for a user into JSON files for analysis.
# Usage: fetch_activity.sh <github-username> <output-dir>
set -euo pipefail

USER="${1:?usage: fetch_activity.sh <github-username> <output-dir>}"
OUT="${2:?usage: fetch_activity.sh <github-username> <output-dir>}"
mkdir -p "$OUT"

echo "Fetching profile for $USER..." >&2
gh api "users/$USER" > "$OUT/profile.json"

echo "Fetching public events (last ~90 days, up to 300)..." >&2
gh api "users/$USER/events/public" --paginate 2>/dev/null | jq -s 'add // []' > "$OUT/events.json"

echo "Fetching PRs authored by $USER (search API, up to 1000)..." >&2
gh api "search/issues?q=author:${USER}+is:pr&per_page=100&sort=created&order=asc" --paginate 2>/dev/null \
  | jq -s '{total_count: (.[0].total_count // 0), items: (map(.items) | add // [])}' > "$OUT/prs.json"

echo "Fetching issues authored by $USER (search API, up to 1000)..." >&2
gh api "search/issues?q=author:${USER}+is:issue&per_page=100&sort=created&order=asc" --paginate 2>/dev/null \
  | jq -s '{total_count: (.[0].total_count // 0), items: (map(.items) | add // [])}' > "$OUT/issues.json"

echo "Fetching issue/PR comments by $USER (search API, up to 1000)..." >&2
gh api "search/issues?q=commenter:${USER}&per_page=100&sort=updated&order=desc" --paginate 2>/dev/null \
  | jq -s '{total_count: (.[0].total_count // 0), items: (map(.items) | add // [])}' > "$OUT/comments.json"

echo "Done. Data written to $OUT/" >&2
