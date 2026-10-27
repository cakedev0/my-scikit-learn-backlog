#!/usr/bin/env python3
"""Dump a GitHub PR's discussion context as markdown, for review prep.

Usage: fetch_pr_context.py <pr-number-or-url> <outdir> [--repo owner/name]

Writes into <outdir>:
  pr.md            metadata, description, changed files, commits
  discussion.md    conversation comments, reviews and inline review threads
                   (with resolved/outdated status), in chronological order
  refs/<N>.md      every issue/PR referenced from the description or the
                   discussion (title, state, body, comments)
  refs.txt         one line per reference: number, kind, state, title

Requires an authenticated `gh`.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number title url state isDraft createdAt mergedAt
      author { login }
      baseRefName headRefName
      additions deletions changedFiles
      labels(first: 20) { nodes { name } }
      milestone { title }
      reviewDecision
      body
      closingIssuesReferences(first: 20) { nodes { number } }
      files(first: 100) { nodes { path additions deletions } }
      commits(last: 100) {
        totalCount
        nodes { commit { oid messageHeadline committedDate } }
      }
      comments(first: 100) {
        pageInfo { hasNextPage }
        nodes { author { login } createdAt body url }
      }
      reviews(first: 100) {
        pageInfo { hasNextPage }
        nodes { author { login } state submittedAt body }
      }
      reviewThreads(first: 100) {
        pageInfo { hasNextPage }
        nodes {
          isResolved isOutdated path line originalLine
          resolvedBy { login }
          comments(first: 50) {
            nodes { author { login } createdAt body diffHunk }
          }
        }
      }
    }
  }
}
"""

REF_RE = re.compile(
    r"(?:(?<![\w/])#|github\.com/(?P<repo>[\w.-]+/[\w.-]+)/(?:issues|pull)/)"
    r"(?P<num>\d{2,6})\b"
)


def gh(*args):
    out = subprocess.run(["gh", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return out.stdout


def login(node):
    return (node or {}).get("login") or "ghost"


def parse_target(target, repo):
    m = re.match(r"https://github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)", target)
    if m:
        return m.group(1), int(m.group(2))
    return repo, int(target.lstrip("#"))


def collect_refs(texts, repo, own_number):
    refs = set()
    for text in texts:
        for m in REF_RE.finditer(text or ""):
            ref_repo = m.group("repo")
            if ref_repo and ref_repo.lower() != repo.lower():
                continue
            num = int(m.group("num"))
            if num != own_number:
                refs.add(num)
    return sorted(refs)


def fetch_ref(repo, num):
    issue = json.loads(gh("api", f"repos/{repo}/issues/{num}"))
    comments = json.loads(
        gh("api", "--paginate", f"repos/{repo}/issues/{num}/comments?per_page=100")
    )
    kind = "PR" if "pull_request" in issue else "issue"
    if kind == "PR" and issue["pull_request"].get("merged_at"):
        state = "merged"
    else:
        state = issue["state"]
    lines = [
        f"# {kind} #{num}: {issue['title']}",
        "",
        f"- url: {issue['html_url']}",
        f"- state: {state} | author: {login(issue['user'])} "
        f"| created: {issue['created_at'][:10]}",
        f"- labels: {', '.join(lb['name'] for lb in issue['labels']) or '-'}",
        "",
        "## Body",
        "",
        issue.get("body") or "_(empty)_",
        "",
        f"## Comments ({len(comments)})",
    ]
    for c in comments:
        lines += ["", f"### {login(c['user'])} ({c['created_at'][:10]})", "", c["body"]]
    return kind, state, issue["title"], "\n".join(lines) + "\n"


def main():
    args = sys.argv[1:]
    repo = "scikit-learn/scikit-learn"
    if "--repo" in args:
        i = args.index("--repo")
        repo = args[i + 1]
        del args[i : i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    repo, number = parse_target(args[0], repo)
    outdir = Path(args[1])
    (outdir / "refs").mkdir(parents=True, exist_ok=True)

    owner, name = repo.split("/")
    data = json.loads(
        gh(
            "api", "graphql", "-f", f"query={QUERY}",
            "-f", f"owner={owner}", "-f", f"name={name}", "-F", f"number={number}",
        )
    )
    pr = data["data"]["repository"]["pullRequest"]

    # pr.md
    closing = [n["number"] for n in pr["closingIssuesReferences"]["nodes"]]
    commits = pr["commits"]["nodes"]
    lines = [
        f"# PR #{pr['number']}: {pr['title']}",
        "",
        f"- url: {pr['url']}",
        f"- author: {login(pr['author'])} | state: {pr['state']}"
        f"{' (draft)' if pr['isDraft'] else ''} | review decision: "
        f"{pr['reviewDecision'] or '-'}",
        f"- created: {pr['createdAt'][:10]} | merged: {(pr['mergedAt'] or '-')[:10]}",
        f"- {pr['headRefName']} -> {pr['baseRefName']} | +{pr['additions']} "
        f"-{pr['deletions']} in {pr['changedFiles']} files",
        f"- labels: {', '.join(n['name'] for n in pr['labels']['nodes']) or '-'}",
        f"- milestone: {(pr['milestone'] or {}).get('title', '-')}",
        f"- closes: {', '.join(f'#{n}' for n in closing) or '-'}",
        "",
        "## Description",
        "",
        pr["body"] or "_(empty)_",
        "",
        "## Changed files",
        "",
    ]
    lines += [
        f"- {f['path']} (+{f['additions']} -{f['deletions']})"
        for f in pr["files"]["nodes"]
    ]
    lines += ["", f"## Commits ({pr['commits']['totalCount']}, last 100 shown)", ""]
    lines += [
        f"- {c['commit']['committedDate'][:10]} {c['commit']['oid'][:10]} "
        f"{c['commit']['messageHeadline']}"
        for c in commits
    ]
    (outdir / "pr.md").write_text("\n".join(lines) + "\n")

    # discussion.md: merge conversation comments, review summaries and inline
    # threads into one timeline (threads are placed at their first comment).
    events = []
    for c in pr["comments"]["nodes"]:
        events.append((c["createdAt"], f"### Comment by {login(c['author'])} "
                       f"({c['createdAt'][:10]})\n\n{c['body']}"))
    for r in pr["reviews"]["nodes"]:
        if not r["body"] and r["state"] == "COMMENTED":
            continue  # empty wrapper around inline comments
        events.append((r["submittedAt"] or "", f"### Review by {login(r['author'])}: "
                       f"{r['state']} ({(r['submittedAt'] or '')[:10]})\n\n"
                       f"{r['body'] or '_(no summary)_'}"))
    for t in pr["reviewThreads"]["nodes"]:
        comments = t["comments"]["nodes"]
        if not comments:
            continue
        status = []
        if t["isResolved"]:
            status.append(f"RESOLVED by {login(t['resolvedBy'])}")
        else:
            status.append("UNRESOLVED")
        if t["isOutdated"]:
            status.append("outdated")
        line = t["line"] or t["originalLine"]
        hunk = "\n".join(comments[0]["diffHunk"].splitlines()[-6:])
        parts = [
            f"### Inline thread on {t['path']}:{line} [{', '.join(status)}]",
            "",
            "```diff",
            hunk,
            "```",
        ]
        for c in comments:
            parts += ["", f"**{login(c['author'])}** ({c['createdAt'][:10]}):", "",
                      c["body"]]
        events.append((comments[0]["createdAt"], "\n".join(parts)))
    events.sort(key=lambda e: e[0])

    truncated = [k for k in ("comments", "reviews", "reviewThreads")
                 if pr[k]["pageInfo"]["hasNextPage"]]
    header = [f"# Discussion on PR #{number}", ""]
    if truncated:
        header += [f"**Truncated to the first 100 {', '.join(truncated)}.** "
                   f"Use `gh api` to fetch the rest if needed.", ""]
    n_unresolved = sum(
        not t["isResolved"] for t in pr["reviewThreads"]["nodes"]
    )
    header += [f"{len(events)} entries, {n_unresolved} unresolved inline threads.", ""]
    (outdir / "discussion.md").write_text(
        "\n".join(header) + "\n\n".join(e[1] for e in events) + "\n"
    )

    # Referenced issues / PRs.
    texts = [pr["body"]]
    texts += [c["body"] for c in pr["comments"]["nodes"]]
    texts += [r["body"] for r in pr["reviews"]["nodes"]]
    texts += [c["body"] for t in pr["reviewThreads"]["nodes"]
              for c in t["comments"]["nodes"]]
    refs = sorted(set(collect_refs(texts, repo, number)) | set(closing))
    ref_lines = []
    for num in refs:
        try:
            kind, state, title, md = fetch_ref(repo, num)
        except RuntimeError as exc:
            ref_lines.append(f"#{num}\t?\t?\tfetch failed: {exc}")
            continue
        (outdir / "refs" / f"{num}.md").write_text(md)
        tag = " (closed by this PR)" if num in closing else ""
        ref_lines.append(f"#{num}\t{kind}\t{state}\t{title}{tag}")
    (outdir / "refs.txt").write_text("\n".join(ref_lines) + "\n")

    print(f"Wrote {outdir}/pr.md, discussion.md, refs.txt and {len(refs)} refs/*.md")
    print(f"{len(events)} discussion entries, {n_unresolved} unresolved threads")
    if truncated:
        print(f"WARNING: truncated {', '.join(truncated)} to 100")


if __name__ == "__main__":
    main()
