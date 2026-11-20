#!/usr/bin/env python3
"""Plot the one-off pip/uv installs of PyPI packages from the download logs.

Most PyPI downloads come from mirrors and scrapers. This script keeps the
downloads made by an installer (pip, uv, ...) and groups them by client. The
logs have no IP address, so a client is a fingerprint: country, installer, OS,
Python version, glibc version and CI flag. A client that downloaded more than
one file on some day is counted as automated (CI, scraper, or someone testing
several versions), and the downloads of the other clients are the one-off
installs.

Two people with the same fingerprint are counted as one client. So the one-off
installs only tell something for a small package (up to a few installs a day).
On a popular one, the common fingerprints have many downloads a day and count
as automated: the one-off installs are then the installs from rare setups, a
small and biased sample of the people who install the package.

A package with up to 100 one-off installs gets one point per install in the
plot. Above that, the plot shows the downloads and the one-off installs per
day. With `--report`, the plots of several packages go to a directory, with a
README.md that has a table of the numbers and all the plots.

The data comes from the public ClickHouse instance behind
https://clickpy.clickhouse.com, which does all the counting. Needs matplotlib.

    scripts/pypi-installs/plot_installs.py sklearn-contrib-emeans
    scripts/pypi-installs/plot_installs.py some-package --since 2026-11-01 -o out.png
    scripts/pypi-installs/plot_installs.py scikit-learn xgboost --report some-dir
"""

import argparse
import datetime
import json
import pathlib
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
from packaging.version import Version  # noqa: E402

CLICKHOUSE_URL = "https://sql-clickhouse.clickhouse.com/?user=demo"
INSTALLERS = ("pip", "uv", "poetry", "pdm", "pixi")
CLIENT = "country, installer, system, python, glibc, ci"
# Above this number of one-off installs, one point per install is unreadable.
MAX_POINTS = 100
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
        with urllib.request.urlopen(request, timeout=90) as response:
            return [json.loads(line) for line in response.read().splitlines() if line]
    except urllib.error.HTTPError as error:
        message = error.read().decode(errors="replace").strip()
        raise SystemExit(f"Query failed: {message}") from None


def classified(project, since, until):
    """SQL of one row per (day, client, version), with the one-off flag."""
    installers = ", ".join(f"'{name}'" for name in INSTALLERS)
    per_version = (
        "SELECT toString(date) AS day, country_code AS country, installer,"
        " system, python_minor AS python, libc.version AS glibc,"
        " toString(ci) AS ci, version, count() AS n"
        " FROM pypi.pypi"
        f" WHERE project = '{project}' AND date >= '{since}' AND date <= '{until}'"
        f" AND installer IN ({installers})"
        " GROUP BY ALL"
    )
    per_day = (
        f"SELECT *, sum(n) OVER (PARTITION BY day, {CLIENT}) AS day_n"
        f" FROM ({per_version})"
    )
    return (
        f"SELECT *, max(day_n) OVER (PARTITION BY {CLIENT}) = 1 AND ci != 'true'"
        f" AS is_one_off FROM ({per_day})"
    )


def fetch_summary(project, since, until):
    """The numbers of one package over the period, or None without download."""
    (row,) = query(
        "SELECT min(day) AS first, max(day) AS last, sum(n) AS downloads,"
        " sumIf(n, ci = 'true') AS in_ci, sumIf(n, installer = 'uv') AS uv,"
        f" uniqExact({CLIENT}) AS clients,"
        f" uniqExactIf(({CLIENT}), is_one_off) AS one_off_clients,"
        " sumIf(n, is_one_off) AS one_off"
        f" FROM ({classified(project, since, until)})"
    )
    summary = {key: int(value) for key, value in row.items() if key[-2:] != "st"}
    if not summary["downloads"]:
        return None
    summary["first"] = datetime.date.fromisoformat(row["first"])
    summary["last"] = datetime.date.fromisoformat(row["last"])
    # All the downloads, with the mirrors and scrapers.
    (row,) = query(
        f"SELECT count() AS n FROM pypi.pypi WHERE project = '{project}'"
        f" AND date >= '{since}' AND date <= '{until}'"
    )
    summary["all"] = int(row["n"])
    return summary


def fetch_top_automated(project, since, until, limit=20):
    return query(
        f"SELECT {CLIENT}, sum(n) AS n FROM ({classified(project, since, until)})"
        f" WHERE NOT is_one_off GROUP BY {CLIENT} ORDER BY n DESC LIMIT {limit}"
    )


def fetch_one_off(project, since, until):
    """One row per one-off install."""
    return query(
        f"SELECT day, {CLIENT}, version FROM ({classified(project, since, until)})"
        " WHERE is_one_off ORDER BY day, version"
    )


def fetch_daily(project, since, until):
    """Downloads and one-off installs per (day, OS)."""
    return query(
        "SELECT day, system, sum(n) AS downloads, sumIf(n, is_one_off) AS one_off"
        f" FROM ({classified(project, since, until)})"
        " GROUP BY day, system ORDER BY day"
    )


def os_name_of(system):
    return OS_NAMES.get(system, system or "unknown")


def short_count(value, _=None):
    for unit, size in (("M", 1e6), ("k", 1e3)):
        if value >= size:
            return f"{value / size:g}{unit}"
    return f"{value:g}"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def set_day_ticks(ax, days):
    every = max(1, len(days) // 12)
    ax.set_xticks(
        range(0, len(days), every),
        [f"{day:%b} {day.day}" for day in days[::every]],
    )


def plot_daily(daily, summary, project, output):
    """Bars per day: all the installer downloads, then the one-off installs by OS."""
    since, until = summary["first"], summary["last"]
    days = [since + datetime.timedelta(days=i) for i in range((until - since).days + 1)]
    downloads = [0] * len(days)
    one_off = {os_name: [0] * len(days) for os_name in (*OS_STYLES, "Other")}
    for row in daily:
        x = days.index(datetime.date.fromisoformat(row["day"]))
        downloads[x] += int(row["downloads"])
        os_name = os_name_of(row["system"])
        os_name = os_name if os_name in OS_STYLES else "Other"
        one_off[os_name][x] += int(row["one_off"])

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 6.4), dpi=150, facecolor=SURFACE, sharex=True
    )
    xs = range(len(days))
    bar_style = {"width": 0.8, "edgecolor": SURFACE, "linewidth": 1}
    top.bar(xs, downloads, color=OTHER_OS_STYLE[0], **bar_style)
    top.set_title(
        f"The {summary['downloads']:,} downloads of {project} by an installer,"
        f" {since:%b} {since.day} to {until:%b} {until.day}, {until.year}",
        loc="left",
        fontsize=11,
        color=INK,
        pad=10,
    )

    base = [0] * len(days)
    for os_name, counts in one_off.items():
        if not sum(counts):
            continue
        color = OS_STYLES.get(os_name, OTHER_OS_STYLE)[0]
        label = f"{os_name} ({sum(counts):,})"
        bottom.bar(xs, counts, bottom=base, color=color, label=label, **bar_style)
        base = [low + count for low, count in zip(base, counts)]
    bottom.set_title(
        f"Of which {summary['one_off']:,} one-off installs,"
        f" from {summary['one_off_clients']:,} of the {summary['clients']:,} clients",
        loc="left",
        fontsize=11,
        color=INK,
        pad=26,
    )
    bottom.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncols=4,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        borderaxespad=0.2,
    )
    for ax in (top, bottom):
        style_axes(ax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(short_count))
        ax.set_ylabel("Per day", color=MUTED, fontsize=9)
    set_day_ticks(bottom, days)
    bottom.set_xlim(-0.6, len(days) - 0.4)
    fig.tight_layout(h_pad=2)
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)


def plot_points(one_off, project, since, until, output):
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

    os_counts = Counter()
    for (day, version), cell in cells.items():
        x = days.index(datetime.date.fromisoformat(day))
        base = versions.index(version)
        for i, row in enumerate(cell):
            y = base + (i - (len(cell) - 1) / 2) * step
            os_name = os_name_of(row["system"])
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
    set_day_ticks(ax, days)
    ax.set_xlim(-0.5, len(days) - 0.1)
    ax.set_ylabel("Version downloaded", color=MUTED, fontsize=9)
    style_axes(ax)
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
    plt.close(fig)


def describe(client):
    details = "{country} {installer} {system} py{python}".format(**client)
    details += f" glibc {client['glibc']}" if client["glibc"] else ""
    return details + (" (CI)" if client["ci"] == "true" else "")


def run(project, since, until, output):
    """Print the numbers of one package, plot it and return its summary."""
    summary = fetch_summary(project, since, until)
    if summary is None:
        print(f"No download of {project} by an installer in this period.")
        return None
    # The plot covers the days with data, not the whole requested period.
    first, last = summary["first"], summary["last"]
    automated = summary["clients"] - summary["one_off_clients"]
    print(f"{project}, {first} to {last}")
    print(f"Downloads: {summary['all']:,}")
    print(
        f"Downloads by an installer: {summary['downloads']:,}"
        f" ({summary['in_ci']:,} in CI, {summary['uv']:,} by uv)"
    )
    top = fetch_top_automated(project, since, until)
    shown = f", the top {len(top)}" if len(top) < automated else ""
    print(f"\nAutomated or recurring clients ({automated:,}{shown}):")
    for client in top:
        print(f"  {int(client['n']):11,d}  {describe(client)}")
    print(
        f"\nOne-off installs ({summary['one_off']:,},"
        f" from {summary['one_off_clients']:,} clients):"
    )
    if summary["one_off"] > MAX_POINTS:
        print("  too many to list, see the plot")
        plot_daily(fetch_daily(project, since, until), summary, project, output)
    elif summary["one_off"]:
        one_off = fetch_one_off(project, since, until)
        for row in one_off:
            print(f"  {row['day']}  {row['version']:9s} {describe(row)}")
        plot_points(one_off, project, first, last, output)
    else:
        return summary
    summary["plot"] = output
    print(f"\nPlot written to {output}\n")
    return summary


def write_report(summaries, since, until, directory):
    lines = [
        f"# PyPI installs, {since} to {until}",
        "",
        "> [!WARNING]",
        "> This is one possible way to read the PyPI download logs, to try to get",
        "> some insights. The clients are guessed from coarse fingerprints, and the",
        "> split between automated and one-off installs is a heuristic. These",
        "> numbers might not be representative of anything, and they are not counts",
        "> of users. They only cover PyPI: no conda-forge and no other index.",
        "",
        "Made with `scripts/pypi-installs/plot_installs.py`, from the public",
        "ClickHouse instance behind https://clickpy.clickhouse.com:",
        "",
        "- **Downloads**: all the downloads of the package, mostly mirrors,",
        "  scrapers and CI.",
        f"- **By an installer**: the downloads made by {', '.join(INSTALLERS)}.",
        "  **In CI** and **By uv** are shares of these downloads.",
        "- **Clients**: the logs have no IP address, so a client is a fingerprint",
        "  (country, installer, OS, Python version, glibc version and CI flag).",
        "  Two people with the same fingerprint are one client.",
        "- **One-off installs**: the downloads of the clients that are not in CI",
        "  and never downloaded more than one file of the package in a day.",
        "  **One-off clients** is the number of these clients.",
        "",
        "On a popular package, the common fingerprints (say pip on Linux with",
        "Python 3.12 in the US) have many downloads a day and count as automated.",
        "The one-off installs are then the installs from rare setups: a small and",
        "biased sample, which can't be compared between a small package and a",
        "popular one.",
        "",
        (
            "| Package | Downloads | By an installer | In CI | By uv | Clients"
            " | One-off clients | One-off installs |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for project, summary in summaries.items():
        downloads = summary["downloads"]
        lines.append(
            f"| [{project}](#{project}) | {summary['all']:,} | {downloads:,}"
            f" | {summary['in_ci'] / downloads:.0%} | {summary['uv'] / downloads:.0%}"
            f" | {summary['clients']:,} | {summary['one_off_clients']:,}"
            f" | {summary['one_off']:,} |"
        )
    lines += [
        "",
        f"With up to {MAX_POINTS} one-off installs, the plot has one point per",
        "install. Above that, it shows the downloads by an installer and the",
        "one-off installs per day.",
    ]
    for project, summary in summaries.items():
        lines += ["", f"## {project}", ""]
        if "plot" in summary:
            name = pathlib.Path(summary["plot"]).name
            lines.append(f'<img src="{name}" alt="Installs of {project}" width="900">')
        else:
            lines.append("No one-off install.")
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Report written to {directory / 'README.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("projects", nargs="+", help="names of the packages on PyPI")
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
        "-o",
        "--output",
        help="image to write, for one package (default: <project>_installs.png)",
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        help="directory to write the plots and a README.md with all the numbers",
    )
    args = parser.parse_args()
    if args.output and (args.report or len(args.projects) > 1):
        parser.error("--output only works with one package and without --report")

    summaries = {}
    for name in args.projects:
        # The name goes in the SQL query: only accept a valid package name,
        # and normalize it like PyPI does.
        if not re.fullmatch(r"[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?", name):
            parser.error(f"{name!r} is not a valid package name")
        project = re.sub(r"[-_.]+", "-", name).lower()
        output = args.output or f"{project}_installs.png"
        if args.report:
            args.report.mkdir(parents=True, exist_ok=True)
            output = str(args.report / f"{project}.png")
        summary = run(project, args.since, args.until, output)
        if summary:
            summaries[project] = summary
    if not summaries:
        raise SystemExit(1)
    if args.report:
        write_report(summaries, args.since, args.until, args.report)


if __name__ == "__main__":
    main()
