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
from datetime import date, datetime, timedelta
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
  timelineItems(itemTypes: [REVIEW_REQUESTED_EVENT], last: 10) {
    nodes {
      ... on ReviewRequestedEvent {
        createdAt
        actor { login }
        requestedReviewer { ... on User { login } }
      }
    }
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
    """URLs that were once on the list and no longer are.

    -> {url: {"time", "commit", "kind"}}, where kind is "done" or "drop" when
    a `done:` or `drop:` line of the removing commit names the thread's
    number, "unknown" otherwise.
    """
    log = run(
        "git", "log", "--reverse", "-p", "--format=COMMIT %H %cI", "--",
        "TODO.md", "todo/", cwd=BACKLOG,
    )  # fmt: skip
    removed, commit = {}, None
    for line in log.splitlines():
        if line.startswith("COMMIT "):
            commit = tuple(line.split()[1:3])
        elif line.startswith("-") and not line.startswith("---"):
            for url in URL_RE.findall(line):
                removed[url] = commit
        elif line.startswith("+") and not line.startswith("+++"):
            for url in URL_RE.findall(line):
                removed.pop(url, None)

    messages = {}
    result = {}
    for url, (sha, time) in removed.items():
        if url in current:
            continue
        if sha not in messages:
            messages[sha] = run("git", "show", "-s", "--format=%B", sha, cwd=BACKLOG)
        number = url.rsplit("/", 1)[1]
        kinds = {
            line.split(":", 1)[0]
            for line in messages[sha].splitlines()
            if line.startswith(("done:", "drop:")) and re.search(rf"#{number}\b", line)
        }
        kind = kinds.pop() if len(kinds) == 1 else "unknown"
        result[url] = {"time": time, "commit": sha[:7], "kind": kind}
    return result


def last_sync_date():
    out = run("git", "log", "-1", "--grep=^sync:", "--format=%cs", cwd=BACKLOG)
    return out.strip() or str(date.today() - timedelta(days=30))


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


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

    # The last time a review was requested from me, and by whom.
    requests = [
        e
        for e in (node["timelineItems"]["nodes"] if pr else [])
        if (e.get("requestedReviewer") or {}).get("login") == user
    ]
    request = max(requests, key=lambda e: e["createdAt"], default=None)
    request_new = (
        "review-requested" in roles
        and request is not None
        and (my_last is None or request["createdAt"] > my_last)
    )

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
        if "review-requested" in roles and (others_new or request_new):
            self_requested = request and (request["actor"] or {}).get("login") == user
            reasons.append("self-requested" if self_requested else "review-requested")
        if any(f"@{user}".lower() in (body or "").lower() for _, _, body in others_new):
            reasons.append("mentioned")
        if others_new and (mine or my_last is not None):
            who = sorted({who for _, who, _ in others_new})
            reasons.append("new-activity:" + ",".join(who))
        if "assignee" in roles and my_last is None:
            reasons.append("assigned")
        move = "me" if reasons else "them"

    # Since when the ball is in my court: the first event after my last
    # activity, or after the review request, that put it there. Comments made
    # before a review request weren't addressed to me. Unknown for a red CI
    # or conflicts.
    floor = request["createdAt"] if request_new else my_last
    waiting = [t for t, _, _ in others_new if floor is None or t > floor]
    if request_new:
        waiting.append(request["createdAt"])
    waiting_since = min(waiting)[:10] if move == "me" and waiting else None

    # Events that hand the ball to me, to tell whether a thread came back
    # after I removed it from the list.
    to_me = [t for t, _, _ in others] + ([request["createdAt"]] if request else [])

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
        "waiting_since": waiting_since,
        "review_requested": {
            "date": request["createdAt"][:10],
            "by": (request["actor"] or {}).get("login"),
        }
        if request
        else None,
        "review_decision": node.get("reviewDecision"),
        "ci": rollup,
        "mergeable": node.get("mergeable"),
        "created": node["createdAt"][:10],
        "_to_me": to_me,
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
        to_me = t.pop("_to_me")
        if t["removed"]:
            removed_at = parse_time(t["removed"]["time"])
            t["removed"]["back"] = t["move"] != "closed" and any(
                parse_time(time) > removed_at for time in to_me
            )
        threads.append(t)
    order = {"me": 0, "them": 1, "closed": 2}
    threads.sort(key=lambda t: (not t["tracked"], order[t["move"]], t["updated"]))

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "threads.json").write_text(json.dumps(threads, indent=1))

    print(f"since {since}: {len(threads)} threads, {len(tracked)} tracked")
    print("tracked\tmove\treasons\twaiting_since\tupdated\turl\ttitle")
    for t in threads:
        removed = t["removed"]
        flag = "yes" if t["tracked"] else "no"
        if removed:
            flag = f"{removed['kind']}{'+back' if removed['back'] else ''}"
        reasons = " ".join(t["reasons"]) or "-"
        print(
            f"{flag}\t{t['move']}\t{reasons}\t{t['waiting_since'] or '-'}"
            f"\t{t['updated']}\t{t['url']}\t{t['title']}"
        )


if __name__ == "__main__":
    main()
