#!/usr/bin/env python3
"""Condense my recent Claude Code sessions, for /todo-sync.

Usage: read_transcripts.py <outdir> [--since YYYY-MM-DD]

Reads the transcripts of ~/.claude/projects/*/*.jsonl with activity since
--since (default: the date of the last `sync:` commit, or 30 days ago), and
keeps only the sessions started in a public GitHub repo of the scikit-learn
world (RELATED below, forks included). Sessions started outside a git repo
(e.g. from ~), in a private repo or in an unrelated one (personal projects)
are skipped, and so is the running /todo-sync session.

For each kept session, writes <outdir>/transcripts/<session-id>.md with the
title, my prompts, the last summaries and the last assistant messages, and
prints one row per session: id, last activity, repo, prompts, title, and the
end of the last assistant message (where open questions usually are).

Only reads local files, plus `gh api repos/...` to check that a repo is public.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BACKLOG = Path(__file__).resolve().parents[2]
PROJECTS = Path.home() / ".claude" / "projects"

# Canonical repos (the source repo for a fork) whose sessions are read.
RELATED = {
    "scikit-learn/scikit-learn",
    "scikit-learn/blog",
    "probabl-ai/scikit-learn-benchmarks",
    "cakedev0/my-scikit-learn-backlog",
    "joblib/joblib",
    "joblib/loky",
    "joblib/threadpoolctl",
    "uxlfoundation/scikit-learn-intelex",
    "uxlfoundation/oneDAL",
    "cython/cython",
    "scipy/scipy",
    "numpy/numpy",
    "python/cpython",
    "OpenMathLib/OpenBLAS",
    "IntelPython/dpnp",
    "rapidsai/raft",
}

REMOTE_RE = re.compile(r"github\.com[:/]([\w.-]+/[\w.-]+?)(?:\.git)?$")
PROMPT_CHARS = 600
ASSISTANT_CHARS = 2500
ROW_TAIL_CHARS = 200


def run(*cmd, cwd=None):
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return res.stdout.strip() if res.returncode == 0 else None


def last_sync_date():
    out = run("git", "log", "-1", "--grep=^sync:", "--format=%cs", cwd=BACKLOG)
    return out or (date.today() - timedelta(days=30)).isoformat()


_repo_cache = {}


def canonical_public_repo(full_name):
    """Source repo of `full_name` if it is public, else None."""
    if full_name not in _repo_cache:
        out = run(
            "gh", "api", f"repos/{full_name}",
            "--jq", "[.private, (.source.full_name // .full_name)] | @tsv",
        )
        result = None
        if out:
            private, source = out.split("\t")
            if private == "false":
                result = source
        _repo_cache[full_name] = result
    return _repo_cache[full_name]


def session_repo(cwd):
    """The RELATED repo a session's working directory belongs to, or None."""
    if not cwd or not Path(cwd).is_dir():
        return None
    remotes = run("git", "-C", cwd, "remote", "-v")
    if not remotes:
        return None
    names = []
    for line in remotes.splitlines():
        m = REMOTE_RE.search(line.split()[1])
        if m and m.group(1) not in names:
            names.append(m.group(1))
    for name in names:
        canonical = canonical_public_repo(name)
        if canonical in RELATED:
            return canonical
    return None


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def clean_prompt(text):
    """Keep a slash command as `/name args`, drop the command's expanded body."""
    name = re.search(r"<command-name>(.*?)</command-name>", text, re.S)
    if name:
        args = re.search(r"<command-args>(.*?)</command-args>", text, re.S)
        return f"{name.group(1).strip()} {args.group(1).strip() if args else ''}".strip()
    return text.strip()


def shorten(text, n, keep="start"):
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n] + " […]" if keep == "start" else "[…] " + text[-n:]


def read_session(path):
    s = {
        "id": path.stem, "cwd": None, "branch": None, "first": None, "last": None,
        "title": None, "prompts": [], "summaries": [], "assistant": [],
        "pr_links": [],
    }
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("isSidechain"):
                continue
            ts = d.get("timestamp")
            if ts:
                s["first"] = min(s["first"] or ts, ts)
                s["last"] = max(s["last"] or ts, ts)
            s["cwd"] = s["cwd"] or d.get("cwd")
            s["branch"] = d.get("gitBranch") or s["branch"]
            kind = d.get("type")
            if kind == "ai-title":
                s["title"] = d.get("aiTitle")
            elif kind == "pr-link":
                url = d.get("prUrl") or d.get("url")
                if url and url not in s["pr_links"]:
                    s["pr_links"].append(url)
            elif kind == "system" and d.get("subtype") == "away_summary":
                s["summaries"].append((ts, d.get("content", "")))
            elif kind == "user" and (d.get("origin") or {}).get("kind") == "human":
                text = clean_prompt(text_of(d["message"]["content"]))
                if text:
                    s["prompts"].append((ts, text))
            elif kind == "assistant":
                text = text_of(d["message"]["content"]).strip()
                if text:
                    s["assistant"].append((ts, text))
    return s


def write_session(s, repo, outdir):
    lines = [
        f"# {s['title'] or '(untitled)'}",
        "",
        f"- session: {s['id']}",
        f"- repo: {repo}, cwd: {s['cwd']}, branch: {s['branch']}",
        f"- from {s['first'][:16]} to {s['last'][:16]} (UTC)",
    ]
    if s["pr_links"]:
        lines.append(f"- PRs: {' '.join(s['pr_links'])}")
    lines += ["", "## My prompts", ""]
    for ts, text in s["prompts"]:
        lines.append(f"- {ts[:16]}: {shorten(text, PROMPT_CHARS)}")
    if s["summaries"]:
        lines += ["", "## Last summaries", ""]
        for ts, text in s["summaries"][-2:]:
            lines.append(f"- {ts[:16]}: {text.strip()}")
    lines += ["", "## Last assistant messages", ""]
    last_prompt = s["prompts"][-1][0] if s["prompts"] else ""
    # The messages after my last prompt, at least the last two.
    after = [m for m in s["assistant"] if m[0] >= last_prompt]
    for ts, text in (after if len(after) >= 2 else s["assistant"][-2:])[-4:]:
        lines += [f"### {ts[:16]}", "", shorten(text, ASSISTANT_CHARS, keep="end"), ""]
    (outdir / f"{s['id']}.md").write_text("\n".join(lines) + "\n")


def is_running_sync(s):
    """A /todo-sync session active in the last 15 minutes: the one running."""
    recent = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    return (
        bool(s["prompts"])
        and s["prompts"][-1][1].startswith("/todo-sync")
        and s["last"] >= recent[:19]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--since", default=None)
    args = parser.parse_args()
    since = args.since or last_sync_date()
    outdir = args.outdir / "transcripts"
    outdir.mkdir(parents=True, exist_ok=True)

    sessions, skipped = [], {}
    for path in PROJECTS.glob("*/*.jsonl"):
        s = read_session(path)
        if not s["last"] or s["last"][:10] < since or not s["prompts"]:
            continue
        repo = session_repo(s["cwd"])
        if repo is None:
            skipped[path.parent.name] = skipped.get(path.parent.name, 0) + 1
            continue
        sessions.append((s, repo))

    sessions.sort(key=lambda x: x[0]["last"])
    running = next(
        (s["id"] for s, repo in reversed(sessions)
         if repo == "cakedev0/my-scikit-learn-backlog" and is_running_sync(s)),
        None,
    )
    print(f"since {since}: {len(sessions)} sessions, written to {outdir}")
    if skipped:
        print("skipped (not a public related repo): "
              + ", ".join(f"{k} ({v})" for k, v in sorted(skipped.items())))
    print("session\tlast\trepo\tprompts\ttitle\tend of last answer")
    for s, repo in sessions:
        if s["id"] == running:
            continue
        write_session(s, repo, outdir)
        tail = s["assistant"][-1][1] if s["assistant"] else ""
        tail = " ".join(shorten(tail, ROW_TAIL_CHARS, keep="end").split())
        print(f"{s['id']}\t{s['last'][:16]}\t{repo}\t{len(s['prompts'])}"
              f"\t{s['title'] or '-'}\t{tail}")


if __name__ == "__main__":
    sys.exit(main())
