#!/usr/bin/env python3
"""Collect the GitHub threads that may belong on the todo list, for /todo-sync.

Usage: fetch_activity.py <outdir> [--since YYYY-MM-DD] [--user LOGIN]

Searches the open issues and PRs that involve the user and were updated since
--since (authored, commented, mentioned, reviewed), the open review requests
and assignments whatever their age, plus every
GitHub URL already tracked in TODO.md or todo/*.md, and decides for each
whose move it is:

  me     the ball is in the user's court: new comments or reviews from
         others since the user's last activity, a review request not
         answered since the last push, a mention, red CI or conflicts on
         one of the user's PRs.
  them   the user acted last; waiting on others.
  closed merged or closed.

--since defaults to the date of the last `sync:` commit, or 30 days ago.

Writes <outdir>/threads.json (one object per thread) and prints a summary
table. Only reads from GitHub. Requires an authenticated `gh`.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

BACKLOG = Path(__file__).resolve().parents[2]

THREAD_FIELDS = """
__typename
... on PullRequest {
  url title state isDraft createdAt updatedAt
  author { login }
  repository { nameWithOwner }
  reviewDecision mergeable
  assignees(first: 10) { nodes { login } }
  reviewRequests(first: 20) {
    nodes { requestedReviewer { ... on User { login } } }
  }
  comments(last: 30) { nodes { author { __typename login } createdAt body } }
  reviews(last: 30) { nodes { author { __typename login } submittedAt state body } }
  commits(last: 1) {
    nodes { commit { committedDate statusCheckRollup { state } } }
  }
}
... on Issue {
  url title state createdAt updatedAt
  author { login }
  repository { nameWithOwner }
  assignees(first: 10) { nodes { login } }
  comments(last: 30) { nodes { author { __typename login } createdAt body } }
}
"""

SEARCH_QUERY = (
    """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 25, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes { %s }
  }
}
"""
    % THREAD_FIELDS
)

URL_RE = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/\d+")


def run(*args, cwd=None):
    out = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])}: {out.stderr.strip()}")
    return out.stdout


def graphql(query, **variables):
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        if value is not None:
            args += ["-f", f"{key}={value}"]
    return json.loads(run(*args))["data"]


def search(q):
    nodes, cursor = [], None
    while True:
        page = graphql(SEARCH_QUERY, q=q, cursor=cursor)["search"]
        nodes += [n for n in page["nodes"] if n]
        if not page["pageInfo"]["hasNextPage"]:
            return nodes
        cursor = page["pageInfo"]["endCursor"]


def fetch_urls(urls):
    """Fetch threads by URL, 20 per GraphQL query."""
    urls, nodes = sorted(urls), []
    for start in range(0, len(urls), 20):
        chunk = urls[start : start + 20]
        fields = "\n".join(
            f'r{i}: resource(url: "{url}") {{ {THREAD_FIELDS} }}'
            for i, url in enumerate(chunk)
        )
        data = graphql(f"query {{ {fields} }}")
        nodes += [n for n in data.values() if n]
    return nodes


def tracked_urls():
    files = [BACKLOG / "TODO.md", *sorted((BACKLOG / "todo").glob("*.md"))]
    return {
        url
        for path in files
        if path.exists() and path.name != "INSTRUCTIONS.md"
        for url in URL_RE.findall(path.read_text())
    }


def removed_urls(current):
    """URLs that were once on the list and no longer are -> (date, subject)."""
    log = run(
        "git", "log", "--reverse", "-p", "--format=COMMIT %cs %s", "--",
        "TODO.md", "todo/", cwd=BACKLOG,
    )  # fmt: skip
    removed, commit = {}, None
    for line in log.splitlines():
        if line.startswith("COMMIT "):
            commit = line[len("COMMIT ") :].split(" ", 1)
        elif line.startswith("-") and not line.startswith("---"):
            for url in URL_RE.findall(line):
                removed[url] = commit
        elif line.startswith("+") and not line.startswith("+++"):
            for url in URL_RE.findall(line):
                removed.pop(url, None)
    return {
        url: {"date": d, "commit": s}
        for url, (d, s) in removed.items()
        if url not in current
    }


def last_sync_date():
    out = run("git", "log", "-1", "--grep=^sync:", "--format=%cs", cwd=BACKLOG)
    return out.strip() or str(date.today() - timedelta(days=30))


def is_bot(author):
    login = (author or {}).get("login") or ""
    return (author or {}).get("__typename") == "Bot" or login.endswith("[bot]")


def classify(node, user, roles):
    pr = node["__typename"] == "PullRequest"
    author = (node.get("author") or {}).get("login")
    mine = author == user
    events = [(node["createdAt"], author, "")]  # (time, login, body)
    for c in node["comments"]["nodes"]:
        if not is_bot(c["author"]):
            events.append((c["createdAt"], c["author"]["login"], c["body"]))
    if pr:
        for r in node["reviews"]["nodes"]:
            if r["submittedAt"] and not is_bot(r["author"]):
                events.append((r["submittedAt"], r["author"]["login"], r["body"]))
        commits = node["commits"]["nodes"]
        if commits:
            events.append((commits[0]["commit"]["committedDate"], author, ""))

    my_times = [t for t, who, _ in events if who == user]
    my_last = max(my_times, default=None)
    others = [(t, who, body) for t, who, body in events if who != user]
    others_new = [e for e in others if my_last is None or e[0] > my_last]
    others_last = max(others, default=None)

    rollup = None
    if pr and node["commits"]["nodes"]:
        rollup = (node["commits"]["nodes"][0]["commit"]["statusCheckRollup"] or {})
        rollup = rollup.get("state")

    reasons = []
    if node["state"] != "OPEN":
        move = "closed"
    else:
        if pr and mine and rollup in ("FAILURE", "ERROR"):
            reasons.append("ci-failed")
        if pr and mine and node["mergeable"] == "CONFLICTING":
            reasons.append("conflicts")
        if "review-requested" in roles and others_new:
            reasons.append("review-requested")
        if any(f"@{user}".lower() in (body or "").lower() for _, _, body in others_new):
            reasons.append("mentioned")
        if others_new and (mine or my_last is not None):
            who = sorted({who for _, who, _ in others_new})
            reasons.append("new-activity:" + ",".join(who))
        if "assignee" in roles and my_last is None:
            reasons.append("assigned")
        move = "me" if reasons else "them"

    return {
        "url": node["url"],
        "kind": "pr" if pr else "issue",
        "repo": node["repository"]["nameWithOwner"],
        "title": node["title"],
        "author": author,
        "state": node["state"],
        "draft": node.get("isDraft", False),
        "roles": sorted(roles),
        "move": move,
        "reasons": reasons,
        "my_last": (my_last or "")[:10] or None,
        "others_last": (others_last[0][:10] + " " + others_last[1])
        if others_last
        else None,
        "review_decision": node.get("reviewDecision"),
        "ci": rollup,
        "mergeable": node.get("mergeable"),
        "created": node["createdAt"][:10],
        "updated": node["updatedAt"][:10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--since", default=None)
    parser.add_argument("--user", default="cakedev0")
    args = parser.parse_args()
    user, since = args.user, args.since or last_sync_date()

    searches = {
        "author": f"author:{user} is:open updated:>={since}",
        "review-requested": f"review-requested:{user} is:open is:pr",
        "assignee": f"assignee:{user} is:open",
        "involved": f"involves:{user} is:open updated:>={since}",
        "reviewer": f"reviewed-by:{user} is:open is:pr updated:>={since}",
    }
    nodes, roles = {}, {}
    for role, q in searches.items():
        print(f"searching {q!r}...", file=sys.stderr)
        for node in search(q):
            nodes[node["url"]] = node
            roles.setdefault(node["url"], set()).add(role)

    tracked = tracked_urls()
    missing = tracked - nodes.keys()
    print(f"fetching {len(missing)} tracked threads...", file=sys.stderr)
    for node in fetch_urls(missing):
        nodes[node["url"]] = node
    removed = removed_urls(tracked)

    threads = []
    for url, node in nodes.items():
        t = classify(node, user, roles.get(url, set()))
        t["tracked"] = url in tracked
        t["removed"] = removed.get(url)
        threads.append(t)
    order = {"me": 0, "them": 1, "closed": 2}
    threads.sort(key=lambda t: (not t["tracked"], order[t["move"]], t["updated"]))

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "threads.json").write_text(json.dumps(threads, indent=1))

    print(f"since {since}: {len(threads)} threads, {len(tracked)} tracked")
    print("tracked\tmove\treasons\tupdated\turl\ttitle")
    for t in threads:
        flag = "yes" if t["tracked"] else ("removed" if t["removed"] else "no")
        reasons = " ".join(t["reasons"]) or "-"
        print(f"{flag}\t{t['move']}\t{reasons}\t{t['updated']}\t{t['url']}\t{t['title']}")


if __name__ == "__main__":
    main()
