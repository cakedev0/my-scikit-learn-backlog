#!/usr/bin/env python3
"""Plot the one-off pip/uv installs of a PyPI package from the download logs.

Most PyPI downloads come from mirrors and scrapers. This script keeps the
downloads made by an installer (pip, uv, ...) and groups them by client. The
logs have no IP address, so a client is a fingerprint: country, installer, OS,
Python version, glibc version and CI flag. A client that downloaded more than
one file on some day is counted as automated (CI, scraper, or someone testing
several versions), and the downloads of the other clients are the one-off
installs: one point each in the plot.

Two people with the same fingerprint are counted as one client. So this only
tells something for a small package (up to a few installs a day): on a popular
one, almost every fingerprint has several downloads a day and counts as
automated.

The data comes from the public ClickHouse instance behind
https://clickpy.clickhouse.com. Needs matplotlib.

    scripts/pypi-installs/plot_installs.py sklearn-contrib-emeans
    scripts/pypi-installs/plot_installs.py some-package --since 2026-11-01 -o out.png
"""

import argparse
import datetime
import json
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from packaging.version import Version  # noqa: E402

CLICKHOUSE_URL = "https://sql-clickhouse.clickhouse.com/?user=demo"
INSTALLERS = ("pip", "uv", "poetry", "pdm", "pixi")
OS_NAMES = {"Darwin": "macOS"}
# OS: (color, marker). The marker repeats the color for colorblind readers.
OS_STYLES = {
    "Linux": ("#2a78d6", "o"),
    "macOS": ("#eb6834", "s"),
    "Windows": ("#1baf7a", "^"),
}
OTHER_OS_STYLE = ("#8a8984", "D")
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def query(sql):
    request = urllib.request.Request(
        CLICKHOUSE_URL, data=f"{sql} FORMAT JSONEachRow".encode()
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return [json.loads(line) for line in response.read().splitlines() if line]
    except urllib.error.HTTPError as error:
        message = error.read().decode(errors="replace").strip()
        if "TOO_MANY_ROWS" in message:
            # The public instance returns at most 10,000 rows.
            message = (
                "too many downloads for the public ClickHouse instance: take a"
                " shorter period. This script is meant for small packages."
            )
        raise SystemExit(f"Query failed: {message}") from None


def fetch_downloads(project, since, until):
    """One row per (day, client, version), with the number of downloads."""
    installers = ", ".join(f"'{name}'" for name in INSTALLERS)
    return query(
        "SELECT toString(date) AS day, country_code AS country, installer,"
        " system, python_minor AS python, libc.version AS glibc,"
        " toString(ci) AS ci, version, count() AS n"
        " FROM pypi.pypi"
        f" WHERE project = '{project}' AND date >= '{since}' AND date <= '{until}'"
        f" AND installer IN ({installers})"
        " GROUP BY ALL ORDER BY day, version"
    )


def client_of(row):
    return tuple(
        row[key] for key in ("country", "installer", "system", "python", "glibc", "ci")
    )


def split_clients(rows):
    """Return the downloads of the one-off clients and a summary of the others."""
    per_day = Counter()
    for row in rows:
        per_day[client_of(row), row["day"]] += int(row["n"])
    automated = Counter()
    for (client, _), n in per_day.items():
        if n > 1 or client[-1] == "true":
            automated[client] += 0
    one_off = []
    for row in rows:
        client = client_of(row)
        if client in automated:
            automated[client] += int(row["n"])
        else:
            one_off.append(row)
    return one_off, automated


def plot(one_off, project, since, until, output):
    days = [since + datetime.timedelta(days=i) for i in range((until - since).days + 1)]
    versions = sorted({row["version"] for row in one_off}, key=Version)

    cells = defaultdict(list)
    for row in one_off:
        cells[row["day"], row["version"]].append(row)
    tallest = max((len(cell) for cell in cells.values()), default=1)
    step = min(0.2, 0.8 / tallest)

    width = max(9, 1.25 * len(days) + 1)
    height = max(4.6, 1.3 * len(versions) + 1.2)
    fig, ax = plt.subplots(figsize=(width, height), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    os_counts = Counter()
    for (day, version), cell in cells.items():
        x = days.index(datetime.date.fromisoformat(day))
        base = versions.index(version)
        for i, row in enumerate(cell):
            y = base + (i - (len(cell) - 1) / 2) * step
            os_name = OS_NAMES.get(row["system"], row["system"] or "unknown")
            os_counts[os_name] += 1
            color, marker = OS_STYLES.get(os_name, OTHER_OS_STYLE)
            ax.scatter(
                x - 0.22,
                y,
                s=70,
                color=color,
                marker=marker,
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
            )
            label = f"{row['country']}  {row['installer']}, py{row['python']}"
            ax.text(x - 0.12, y, label, va="center", fontsize=8, color=MUTED)

    for os_name, n in os_counts.most_common():
        color, marker = OS_STYLES.get(os_name, OTHER_OS_STYLE)
        ax.scatter([], [], s=70, color=color, marker=marker, label=f"{os_name} ({n})")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncols=len(os_counts),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        borderaxespad=0.2,
    )

    ax.set_yticks(range(len(versions)), versions)
    ax.set_ylim(-0.6, len(versions) - 0.4)
    every = max(1, len(days) // 12)
    ax.set_xticks(
        range(0, len(days), every),
        [f"{day:%b} {day.day}" for day in days[::every]],
    )
    ax.set_xlim(-0.5, len(days) - 0.1)
    ax.set_ylabel("Version downloaded", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"The {len(one_off)} one-off installs of {project},"
        f" {since:%b} {since.day} to {until:%b} {until.day}, {until.year}",
        loc="left",
        fontsize=11,
        color=INK,
        pad=30,
    )
    fig.tight_layout()
    fig.savefig(output, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("project", help="name of the package on PyPI")
    parser.add_argument(
        "--since",
        type=datetime.date.fromisoformat,
        default=datetime.date.today() - datetime.timedelta(days=30),
        help="first day, YYYY-MM-DD (default: 30 days ago)",
    )
    parser.add_argument(
        "--until",
        type=datetime.date.fromisoformat,
        default=datetime.date.today(),
        help="last day, YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "-o", "--output", help="image to write (default: <project>_installs.png)"
    )
    args = parser.parse_args()

    # The name goes in the SQL query: only accept a valid package name, and
    # normalize it like PyPI does.
    if not re.fullmatch(r"[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?", args.project):
        parser.error(f"{args.project!r} is not a valid package name")
    project = re.sub(r"[-_.]+", "-", args.project).lower()
    output = args.output or f"{project}_installs.png"

    rows = fetch_downloads(project, args.since, args.until)
    if not rows:
        raise SystemExit(f"No download of {project} by an installer in this period.")
    one_off, automated = split_clients(rows)

    # The plot covers the days with data, not the whole requested period.
    first = datetime.date.fromisoformat(rows[0]["day"])
    last = datetime.date.fromisoformat(rows[-1]["day"])

    total = sum(int(row["n"]) for row in rows)
    print(f"Downloads by an installer, {first} to {last}: {total}")
    print(f"\nAutomated or recurring clients ({len(automated)}):")
    for client, n in automated.most_common():
        country, installer, system, python, glibc, ci = client
        details = f"{country} {installer} {system} py{python}"
        details += f" glibc {glibc}" if glibc else ""
        details += " (CI)" if ci == "true" else ""
        print(f"  {n:5d}  {details}")
    print(f"\nOne-off installs ({len(one_off)}):")
    for row in one_off:
        print(
            f"  {row['day']}  {row['version']:9s} {row['country']} {row['installer']}"
            f" {row['system']} py{row['python']}"
        )
    if one_off:
        plot(one_off, project, first, last, output)
        print(f"\nPlot written to {output}")


if __name__ == "__main__":
    main()
