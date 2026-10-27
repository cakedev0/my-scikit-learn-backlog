#!/usr/bin/env python3
"""Compute quantitative low-effort/spam-pattern signals from fetched GitHub
activity JSON (produced by fetch_activity.sh). Prints a single JSON report
to stdout; all interpretation/judgment is left to the caller.
"""
import json
import re
import sys
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse


def load(path):
    with open(path) as f:
        return json.load(f)


def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def repo_of(item):
    # repository_url like https://api.github.com/repos/owner/repo
    parts = urlparse(item["repository_url"]).path.strip("/").split("/")
    return "/".join(parts[-2:])


def owner_of(repo_full_name):
    return repo_full_name.split("/")[0]


def main(outdir):
    profile = load(f"{outdir}/profile.json")
    prs = load(f"{outdir}/prs.json")["items"]
    issues = load(f"{outdir}/issues.json")["items"]
    comments = load(f"{outdir}/comments.json")["items"]

    now = datetime.now(timezone.utc)
    created_at = parse_dt(profile["created_at"])
    account_age_days = max((now - created_at).total_seconds() / 86400, 1e-9)

    prs_sorted = sorted(prs, key=lambda p: p["created_at"])
    repos = [repo_of(p) for p in prs_sorted]
    owners = [owner_of(r) for r in repos]

    # Gaps between consecutive PRs, flagged when they land on different repos/owners
    cross_repo_gaps = []
    for a, b in zip(prs_sorted, prs_sorted[1:]):
        ra, rb = repo_of(a), repo_of(b)
        if ra != rb:
            gap_min = (parse_dt(b["created_at"]) - parse_dt(a["created_at"])).total_seconds() / 60
            cross_repo_gaps.append({
                "gap_minutes": round(gap_min, 1),
                "from": {"repo": ra, "url": a["html_url"], "title": a["title"]},
                "to": {"repo": rb, "url": b["html_url"], "title": b["title"]},
            })
    cross_repo_gaps.sort(key=lambda g: g["gap_minutes"])

    # Outcome breakdown
    merged = sum(1 for p in prs if (p.get("pull_request") or {}).get("merged_at"))
    closed_unmerged = sum(
        1 for p in prs
        if p["state"] == "closed" and not (p.get("pull_request") or {}).get("merged_at")
    )
    open_prs = sum(1 for p in prs if p["state"] == "open")

    close_times_unmerged = []
    for p in prs:
        if p["state"] == "closed" and not (p.get("pull_request") or {}).get("merged_at") and p.get("closed_at"):
            hrs = (parse_dt(p["closed_at"]) - parse_dt(p["created_at"])).total_seconds() / 3600
            close_times_unmerged.append(hrs)

    # Body signals
    def body_len(p):
        return len((p.get("body") or "").strip())

    short_body_prs = [p for p in prs if body_len(p) < 40]

    # Per-day repo diversity (days where PRs opened touch many distinct unrelated repos)
    by_day = {}
    for p, r in zip(prs_sorted, repos):
        day = p["created_at"][:10]
        by_day.setdefault(day, set()).add(r)
    busiest_days = sorted(
        ({"date": d, "distinct_repos": len(rs), "repos": sorted(rs)} for d, rs in by_day.items()),
        key=lambda x: -x["distinct_repos"],
    )[:10]

    comment_repos = sorted({repo_of(c) for c in comments})

    # Duplicate/templated PR bodies reused verbatim across (ideally unrelated) repos
    def normalize(text):
        return re.sub(r"\s+", " ", (text or "")).strip().lower()

    body_groups = defaultdict(list)
    for p in prs:
        norm = normalize(p.get("body"))
        if len(norm) >= 20:  # ignore empty/trivial bodies, already covered above
            body_groups[norm].append(p)
    duplicate_bodies = [
        {
            "reuse_count": len(items),
            "distinct_repos": len({repo_of(p) for p in items}),
            "body_snippet": items[0].get("body", "")[:200],
            "examples": [{"url": p["html_url"], "repo": repo_of(p)} for p in items[:6]],
        }
        for items in body_groups.values()
        if len(items) > 1
    ]
    duplicate_bodies.sort(key=lambda d: -d["reuse_count"])

    report = {
        "user": profile.get("login"),
        "profile": {
            "created_at": profile.get("created_at"),
            "account_age_days": round(account_age_days, 1),
            "public_repos": profile.get("public_repos"),
            "followers": profile.get("followers"),
            "following": profile.get("following"),
            "bio": profile.get("bio"),
            "company": profile.get("company"),
            "hireable": profile.get("hireable"),
        },
        "pr_volume": {
            "total_prs_fetched": len(prs),
            "search_total_count_note": "fetch capped at 1000 by GitHub search API",
            "distinct_repos": len(set(repos)),
            "distinct_owners": len(set(owners)),
            "distinct_repos_per_pr": round(len(set(repos)) / len(prs), 3) if prs else None,
            "prs_per_account_age_day": round(len(prs) / account_age_days, 4) if prs else None,
            "first_pr_at": prs_sorted[0]["created_at"] if prs_sorted else None,
            "last_pr_at": prs_sorted[-1]["created_at"] if prs_sorted else None,
        },
        "outcomes": {
            "merged": merged,
            "closed_unmerged": closed_unmerged,
            "open": open_prs,
            "merge_rate": round(merged / len(prs), 3) if prs else None,
            "median_hours_to_close_when_unmerged": (
                round(statistics.median(close_times_unmerged), 2) if close_times_unmerged else None
            ),
        },
        "body_signals": {
            "short_or_empty_body_count": len(short_body_prs),
            "short_or_empty_body_fraction": round(len(short_body_prs) / len(prs), 3) if prs else None,
            "sample_short_body_prs": [
                {"url": p["html_url"], "title": p["title"], "body": (p.get("body") or "")[:200]}
                for p in short_body_prs[:8]
            ],
        },
        "cross_repo_burstiness": {
            "note": "Gaps between consecutive PRs landing on DIFFERENT repos, sorted ascending. "
                    "Very small gaps across unrelated repos/topics are hard to reconcile with "
                    "genuinely reviewing each codebase.",
            "smallest_gaps": cross_repo_gaps[:15],
            "gaps_under_10min": sum(1 for g in cross_repo_gaps if g["gap_minutes"] < 10),
            "gaps_under_30min": sum(1 for g in cross_repo_gaps if g["gap_minutes"] < 30),
        },
        "busiest_days_by_distinct_repos": busiest_days,
        "duplicate_templated_bodies": duplicate_bodies[:10],
        "issues_authored": {
            "total_fetched": len(issues),
            "distinct_repos": len({repo_of(i) for i in issues}),
        },
        "comments": {
            "total_fetched": len(comments),
            "distinct_repos_commented_on": len(comment_repos),
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: analyze_activity.py <output-dir-from-fetch_activity.sh>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
