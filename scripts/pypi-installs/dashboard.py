#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pyarrow"]
# ///
"""Write a dashboard of who installs a set of PyPI packages, and what.

Raw download counts are dominated by CI, containers and mirrors. This script
splits the downloads made by an installer (pip, uv, ...) by the context the
installer reports (CI flag, OS, Python version, country), and adds the release
dates from PyPI and the conda downloads published by Anaconda. The methods and
their limits are written at the end of the dashboard.

    uv run scripts/pypi-installs/dashboard.py
    uv run scripts/pypi-installs/dashboard.py --until 2026-11-19 -o some-dir

The PyPI logs come from the public ClickHouse instance behind
https://clickpy.clickhouse.com, which allows 60 queries per hour: a run takes
about 20. The query results of the day and the monthly conda files (~15 MB
each) are cached in ~/.cache/pypi-installs.
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import pathlib
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402


@dataclasses.dataclass
class Package:
    name: str
    # Names on PyPI and in the conda channels, when they differ from `name`.
    pypi: tuple = ()
    conda: tuple = ()

    def __post_init__(self):
        self.pypi = self.pypi or (self.name,)
        self.conda = self.conda or (self.name,)


# The first package is the reference of the "per 100" column.
PACKAGES = [
    Package("scikit-learn"),
    # On conda, `xgboost` is a metapackage that pulls `py-xgboost`.
    Package("xgboost", conda=("py-xgboost",)),
    Package("lightgbm"),
    Package("catboost"),
    Package("imbalanced-learn"),
    Package("scikit-survival"),
    Package("scikit-learn-intelex"),
    # One PyPI package per CUDA version.
    Package("cuml", pypi=("cuml-cu11", "cuml-cu12", "cuml-cu13")),
    Package("skore"),
]

CLICKHOUSE_URL = "https://sql-clickhouse.clickhouse.com/?user=demo"
# The public instance stops reading after this number of rows and returns a
# partial result without any error.
MAX_ROWS_READ = 1_000_000_000
CONDA_URL = "https://anaconda-package-data.s3.amazonaws.com/conda/monthly"
CACHE = pathlib.Path.home() / ".cache" / "pypi-installs"
INSTALLERS = ("pip", "uv", "poetry", "pdm", "pixi")
INSTALLER_FILTER = "installer IN ({})".format(", ".join(map(repr, INSTALLERS)))
# The pip and uv logs only have the CI flag since this day.
CI_FLAG_START = datetime.date(2026, 1, 8)
DAYS = 30
CLIENT = "country_code, installer, system, python_minor, libc.version, ci"
SEGMENT_SQL = (
    "multiIf(ci = 'true', 'CI', system = 'Linux', 'Linux', system = 'Windows',"
    " 'Windows', system = 'Darwin', 'macOS', 'Other')"
)
# Segment: color. The order is the stacking order.
SEGMENTS = {
    "CI": "#eda100",
    "Linux": "#2a78d6",
    "macOS": "#eb6834",
    "Windows": "#1baf7a",
    "Other": "#8a8984",
}
SEGMENT_LABELS = {
    "CI": "CI",
    "Linux": "Linux, not CI",
    "macOS": "macOS, not CI",
    "Windows": "Windows, not CI",
    "Other": "Other OS",
}
DESKTOP = ("Windows", "macOS")
# Python version group: color (one hue, light to dark, and gray for unknown).
PYTHONS = {
    "3.9 or older": "#86b6ef",
    "3.10": "#5598e7",
    "3.11": "#2a78d6",
    "3.12": "#1c5cab",
    "3.13": "#104281",
    "3.14 or newer": "#0a2a54",
    "unknown": "#8a8984",
}
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
LINE = "#2a78d6"


# Data


def fetch(url, data=None, retries=3):
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read(), response.headers
        except urllib.error.HTTPError as error:
            message = error.read().decode(errors="replace").strip()
            if "QUOTA_EXCEEDED" in message:
                raise SystemExit(
                    "The hourly quota of the ClickHouse instance is reached. Run"
                    f" again later, the finished queries are cached: {message}"
                ) from None
            if error.code < 500 or attempt == retries - 1:
                raise SystemExit(f"{url} failed: {error.code} {message}") from None
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries - 1:
                raise SystemExit(f"{url} failed: {error}") from None
        time.sleep(5 * (attempt + 1))


def query(sql):
    # The instance allows 60 queries per hour. The results of the day are
    # cached, so that a second run (or a run stopped by the quota) is free.
    key = hashlib.sha256(f"{datetime.date.today()} {sql}".encode()).hexdigest()
    path = CACHE / "queries" / f"{key[:32]}.json"
    if path.exists():
        return json.loads(path.read_text())
    body, headers = fetch(CLICKHOUSE_URL, f"{sql} FORMAT JSONEachRow".encode())
    read = int(json.loads(headers["X-ClickHouse-Summary"])["read_rows"])
    if read > 0.95 * MAX_ROWS_READ:
        raise SystemExit(f"Partial result, the query reads too many rows: {sql}")
    rows = [json.loads(line) for line in body.splitlines() if line]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def where(package, since, until, installers=True):
    projects = ", ".join(map(repr, package.pypi))
    condition = f"project IN ({projects}) AND date >= '{since}' AND date <= '{until}'"
    return condition + (f" AND {INSTALLER_FILTER}" if installers else "")


def fetch_release_dates(package):
    """Version: day of its first upload, and the latest stable version."""
    dates = {}
    for project in package.pypi:
        body, _ = fetch(f"https://pypi.org/pypi/{project}/json")
        for version, files in json.loads(body)["releases"].items():
            days = [file["upload_time_iso_8601"][:10] for file in files]
            if days:
                day = datetime.date.fromisoformat(min(days))
                dates[version] = min(day, dates.get(version, day))
    stable = [v for v in dates if not re.search(r"[a-z]", v)]
    return dates, max(stable or dates, key=dates.get)


def segment_of(ci, system):
    if ci == "true":
        return "CI"
    return {"Linux": "Linux", "Windows": "Windows", "Darwin": "macOS"}.get(
        system, "Other"
    )


def fetch_package(package, until):
    """All the PyPI numbers of one package over the last DAYS days."""
    since = until - datetime.timedelta(days=DAYS - 1)
    previous = since - datetime.timedelta(days=DAYS)
    stats = {"package": package, "since": since, "until": until}

    # The instance allows 60 queries per hour: one scan gives all the
    # breakdowns. It also covers the DAYS days before, for the change.
    keys = {
        "date": "day",
        "python_minor": "python",
        "version": "version",
        "country_code": "country",
        "installer": "installer",
    }
    flags = ", ".join(f"grouping({column}) AS in_{column}" for column in keys)
    recent = f"date >= '{since}' AND {INSTALLER_FILTER}"
    rows = query(
        f"SELECT toString(date) AS day, toString(ci) AS ci_flag, system,"
        f" python_minor, version, country_code, installer, {flags},"
        f" countIf(date >= '{since}') AS every,"
        f" countIf({INSTALLER_FILTER}) AS both_periods,"
        f" countIf({recent}) AS n,"
        f" countIf({recent} AND type = 'sdist') AS sdist,"
        f" countIf({recent} AND ci != 'true' AND system IN ('Windows', 'Darwin'))"
        " AS desktop FROM pypi.pypi"
        f" WHERE {where(package, previous, until, installers=False)}"
        " GROUP BY GROUPING SETS ((date, ci, system), (python_minor), (version),"
        " (country_code), (installer))"
    )
    stats["daily"] = daily = defaultdict(Counter)
    stats["installers"] = installers = Counter()
    stats["pythons"] = pythons = Counter()
    stats["versions"] = versions = {}
    stats["countries"] = countries = {}
    stats["desktop_countries"] = desktop_countries = {}
    stats["previous"] = stats["all"] = stats["sdist"] = 0
    for row in rows:
        # The key of the grouping set of the row is the one with its own flag.
        values = [row[f"in_{column}"] for column in keys]
        (key,) = [k for k, v in zip(keys, values) if values.count(v) == 1]
        n = int(row["n"])
        if key == "date":
            day = datetime.date.fromisoformat(row["day"])
            if day < since:
                stats["previous"] += int(row["both_periods"])
            else:
                daily[day][segment_of(row["ci_flag"], row["system"])] += n
        elif key == "python_minor":
            pythons[python_group(row["python_minor"])] += n
        elif key == "version" and n:
            versions[row["version"]] = n
        elif key == "country_code" and n:
            countries[row["country_code"]] = n
            if int(row["desktop"]):
                desktop_countries[row["country_code"]] = int(row["desktop"])
        elif key == "installer":
            stats["all"] += int(row["every"])
            stats["sdist"] += int(row["sdist"])
            if n:
                installers[row["installer"]] = n
    stats["segments"] = segments = Counter()
    for counts in daily.values():
        segments.update(counts)
    stats["downloads"] = sum(segments.values())
    stats["desktop"] = sum(segments[name] for name in DESKTOP)
    if not stats["downloads"]:
        raise SystemExit(f"No download of {package.name} by an installer.")

    # A client is a fingerprint: the logs have no IP address.
    stats["top_clients"] = top = query(
        "SELECT country_code AS country, installer, system, python_minor AS"
        " python, `libc.version` AS glibc, toString(ci) AS ci, n,"
        f" count() OVER () AS clients FROM (SELECT {CLIENT}, count() AS n"
        f" FROM pypi.pypi WHERE {where(package, since, until)} GROUP BY {CLIENT})"
        " ORDER BY n DESC LIMIT 5"
    )
    stats["clients"] = int(top[0]["clients"])
    stats["release_dates"], stats["latest"] = fetch_release_dates(package)
    return stats


def python_group(python):
    match = re.fullmatch(r"3\.(\d+)", python)
    if not match:
        return "3.9 or older" if python.startswith("2.") else "unknown"
    minor = int(match[1])
    if minor <= 9:
        return "3.9 or older"
    return f"3.{minor}" if minor <= 13 else "3.14 or newer"


def last_months(until, count=12):
    """The last `count` complete months, as their first days."""
    month = until.replace(day=1)
    if (until + datetime.timedelta(days=1)).day != 1:
        month = (month - datetime.timedelta(days=1)).replace(day=1)
    months = [month]
    while len(months) < count:
        months.append((months[-1] - datetime.timedelta(days=1)).replace(day=1))
    return months[::-1]


def fetch_pypi_months(months):
    """Month: {package name: downloads by an installer}."""
    by_project = {name: package for package in PACKAGES for name in package.pypi}
    projects = ", ".join(map(repr, by_project))
    end = (months[-1] + datetime.timedelta(days=31)).replace(day=1)
    # A table aggregated per day: the row limit is not a problem over a year.
    rows = query(
        "SELECT toString(toStartOfMonth(date)) AS month, project, sum(count) AS n"
        " FROM pypi.pypi_downloads_per_day_by_version_by_installer_by_type"
        f" WHERE project IN ({projects}) AND date >= '{months[0]}'"
        f" AND date < '{end}' AND {INSTALLER_FILTER} GROUP BY ALL"
    )
    result = {month: Counter() for month in months}
    for row in rows:
        month = datetime.date.fromisoformat(row["month"])
        result[month][by_project[row["project"]].name] += int(row["n"])
    return result


def fetch_conda_months(months):
    """Month: {package name: {channel: downloads}}, for the published months."""
    by_name = {name: package for package in PACKAGES for name in package.conda}
    CACHE.mkdir(parents=True, exist_ok=True)
    result = {}
    for month in months:
        path = CACHE / f"conda-{month:%Y-%m}.parquet"
        if not path.exists():
            url = f"{CONDA_URL}/{month:%Y}/{month:%Y-%m}.parquet"
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    content = response.read()
            except urllib.error.HTTPError:
                # The file of a month is published a few days after its end.
                continue
            path.write_bytes(content)
        table = pq.read_table(path, columns=["data_source", "pkg_name", "counts"])
        names = table["pkg_name"].cast(pa.string())
        table = table.filter(pc.is_in(names, value_set=pa.array(list(by_name))))
        result[month] = counts = defaultdict(Counter)
        for row in table.to_pylist():
            # The nightly channels are CI of the project itself.
            if not row["data_source"].endswith(("-nightly", "-test")):
                package = by_name[row["pkg_name"]]
                counts[package.name][row["data_source"]] += row["counts"]
    return result


# Numbers


def weekend_ratio(stats):
    """Downloads of an average weekend day over those of an average weekday."""
    totals = {True: [], False: []}
    for day, counts in stats["daily"].items():
        totals[day.weekday() >= 5].append(sum(counts.values()))
    if not totals[True] or not totals[False] or not sum(totals[False]):
        return None
    weekday = sum(totals[False]) / len(totals[False])
    return sum(totals[True]) / len(totals[True]) / weekday


def version_ages(stats):
    """Downloads with a known release date, as (age in days, downloads)."""
    ages = []
    for version, n in stats["versions"].items():
        released = stats["release_dates"].get(version)
        if released:
            ages.append((max(0, (stats["until"] - released).days), n))
    return sorted(ages)


def median_age(ages):
    half, seen = sum(n for _, n in ages) / 2, 0
    for age, n in ages:
        seen += n
        if seen >= half:
            return age
    return None


def months_text(days):
    months = round(days / 30.4)
    return "<1 month" if months < 1 else f"{months} month{'s' * (months > 1)}"


def describe_client(client):
    system = {"Darwin": "macOS"}.get(client["system"], client["system"] or "?")
    details = f"{client['country'] or '?'}, {client['installer']}, {system}"
    details += f", Python {client['python']}" if client["python"] else ""
    details += f", glibc {client['glibc']}" if client["glibc"] else ""
    return details + (", CI" if client["ci"] == "true" else "")


def short(value):
    if value >= 1e6:
        return f"{value / 1e6:.3g}M"
    if value >= 1e4:
        return f"{value / 1e3:.3g}k"
    return f"{value:,.0f}"


def per_100(part, total):
    value = 100 * part / total
    return f"{value:.0f}" if value >= 10 else f"{value:.2g}"


def share(part, total):
    if not total:
        return "n/a"
    value = 100 * part / total
    if 0 < value < 0.05:
        return "<0.1%"
    return f"{value:.0f}%" if value >= 9.5 or value == 0 else f"{value:.1f}%"


def top_shares(counts, count=3):
    total = sum(counts.values())
    top = Counter(counts).most_common(count)
    return ", ".join(f"{name or '?'} {share(n, total)}" for name, n in top) or "n/a"


# Plots


def style_axes(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def short_tick(value, _=None):
    for unit, size in (("M", 1e6), ("k", 1e3)):
        if value >= size:
            return f"{value / size:g}{unit}"
    return f"{value:g}"


def plot_shares(rows, colors, labels, title, output):
    """One 100% stacked bar per package. `rows` is {package name: Counter}."""
    fig, ax = plt.subplots(
        figsize=(9, 0.42 * len(rows) + 1.5), dpi=150, facecolor=SURFACE
    )
    names = list(rows)[::-1]
    left = [0.0] * len(names)
    for key, color in colors.items():
        widths = [
            100 * rows[name][key] / (sum(rows[name].values()) or 1) for name in names
        ]
        if not sum(widths):
            continue
        ax.barh(
            names,
            widths,
            left=left,
            color=color,
            label=labels.get(key, key),
            height=0.7,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        left = [low + width for low, width in zip(left, widths)]
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    style_axes(ax, grid="x")
    ax.tick_params(axis="y", labelcolor=INK)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncols=len(colors),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        borderaxespad=0.2,
        columnspacing=1.2,
        handlelength=1.2,
    )
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=30)
    fig.tight_layout()
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)


def panels(count, columns=3, height=2.1):
    rows = -(-count // columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(9, height * rows + 0.9),
        dpi=150,
        facecolor=SURFACE,
        squeeze=False,
    )
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:count])


def finish_panels(fig, title, output, legend_from=None):
    fig.suptitle(title, x=0.012, ha="left", fontsize=11, color=INK)
    top = 0.985
    if legend_from is not None:
        handles, labels = legend_from.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.005, 1 - 0.32 / fig.get_figheight()),
            ncols=len(labels),
            frameon=False,
            fontsize=9,
            labelcolor=INK,
            columnspacing=1.2,
            handlelength=1.2,
        )
        top = 1 - 0.55 / fig.get_figheight()
    else:
        top = 1 - 0.3 / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=1.5)
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)


def plot_daily(all_stats, output):
    """Downloads per day and segment, one panel per package."""
    fig, axes = panels(len(all_stats))
    for ax, stats in zip(axes, all_stats):
        days = sorted(stats["daily"])
        base = [0] * len(days)
        for segment, color in SEGMENTS.items():
            counts = [stats["daily"][day][segment] for day in days]
            ax.bar(
                range(len(days)),
                counts,
                bottom=base,
                color=color,
                width=0.9,
                label=SEGMENT_LABELS[segment],
            )
            base = [low + count for low, count in zip(base, counts)]
        weekends = [i for i, day in enumerate(days) if day.weekday() == 5]
        ax.set_xticks(weekends, [f"{days[i]:%b} {days[i].day}" for i in weekends])
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(short_tick))
        style_axes(ax)
        ax.tick_params(labelsize=8)
        ax.set_title(stats["package"].name, loc="left", fontsize=9.5, color=INK)
    since, until = all_stats[0]["since"], all_stats[0]["until"]
    finish_panels(
        fig,
        f"Downloads by an installer per day, {since:%b} {since.day} to"
        f" {until:%b} {until.day}, {until.year} (the ticks are Saturdays)",
        output,
        legend_from=axes[0],
    )


def plot_months(series, title, output):
    """One line per package. `series` is {package name: {month: downloads}}."""
    fig, axes = panels(len(series), height=1.9)
    for ax, (name, counts) in zip(axes, series.items()):
        months = sorted(counts)
        values = [counts[month] for month in months]
        ax.plot(range(len(months)), values, color=LINE, linewidth=2)
        ax.set_ylim(0, max(values or [1]) * 1.25 or 1)
        ticks = range(0, len(months), 3)
        ax.set_xticks(ticks, [f"{months[i]:%b %y}" for i in ticks])
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(short_tick))
        style_axes(ax)
        ax.tick_params(labelsize=8)
        ax.set_title(name, loc="left", fontsize=9.5, color=INK)
        if months:
            last = f"{months[-1]:%b %y}: {short(values[-1])}"
            ax.set_title(last, loc="right", fontsize=8, color=MUTED)
    finish_panels(fig, title, output)


# Dashboard


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|---|" + "---:|" * (len(header) - 1))
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return lines


def write_dashboard(all_stats, pypi_months, conda_months, directory):
    since, until = all_stats[0]["since"], all_stats[0]["until"]
    reference = all_stats[0]
    period = f"{since:%b} {since.day} to {until:%b} {until.day}, {until.year}"
    names = [stats["package"].name for stats in all_stats]

    plot_shares(
        {s["package"].name: s["segments"] for s in all_stats},
        SEGMENTS,
        SEGMENT_LABELS,
        f"Where the downloads by an installer come from, {period}",
        directory / "segments.png",
    )
    plot_daily(all_stats, directory / "daily.png")
    plot_shares(
        {s["package"].name: s["pythons"] for s in all_stats},
        PYTHONS,
        {},
        f"Python version of the downloads by an installer, {period}",
        directory / "python.png",
    )
    months = sorted(pypi_months)
    plot_months(
        {name: {m: pypi_months[m][name] for m in months} for name in names},
        "PyPI downloads by an installer per month",
        directory / "months_pypi.png",
    )
    conda_total = {
        month: {name: sum(counts[name].values()) for name in names}
        for month, counts in conda_months.items()
    }
    if conda_months:
        plot_months(
            {name: {m: conda_total[m][name] for m in conda_total} for name in names},
            "conda downloads per month (conda-forge, defaults and rapidsai channels)",
            directory / "months_conda.png",
        )

    lines = [
        "# Who installs scikit-learn and related packages",
        "",
        "> [!WARNING]",
        "> This is one possible way to read the public download logs, to try to",
        "> get some insights. Downloads are not users: one CI job or one Docker",
        "> build is a download each time it runs, and a user with a warm cache",
        "> is none. The splits below rely on what the installers report and on",
        "> heuristics. These numbers might not be representative of anything.",
        "> Read the [methods and limits](#methods-and-limits) before quoting one.",
        "",
        f"PyPI numbers: {period} ({DAYS} days). Made by",
        "`scripts/pypi-installs/dashboard.py`.",
        "",
        "## Summary",
        "",
    ]
    rows = []
    for stats in all_stats:
        downloads = stats["downloads"]
        change = (
            f"{100 * (downloads / stats['previous'] - 1):+.0f}%"
            if stats["previous"]
            else "new"
        )
        rows.append(
            [
                f"[{stats['package'].name}](#{stats['package'].name})",
                short(stats["all"]),
                short(downloads),
                change,
                share(stats["segments"]["CI"], downloads),
                short(stats["desktop"]),
                per_100(stats["desktop"], reference["desktop"]),
            ]
        )
    lines += table(
        [
            "Package",
            "All downloads",
            "By an installer",
            f"vs previous {DAYS} days",
            "In CI",
            "Desktop",
            f"Desktop per 100 {reference['package'].name}",
        ],
        rows,
    )
    lines += [
        "",
        "- **All downloads** is the number usually quoted. It includes mirrors and",
        "  scrapers.",
        f"- **By an installer**: by {', '.join(INSTALLERS)}. All the numbers below",
        "  are about these downloads.",
        "- **In CI**: the installer saw a CI environment variable. Only pip and uv",
        "  report it, so this is a lower bound.",
        "- **Desktop**: downloads from Windows and macOS outside of CI. This is the",
        "  closest thing to people installing the package on their own machine,",
        "  and I think the most comparable number between packages. It misses",
        "  the people on Linux, so it is unfair to Linux-only packages like cuml.",
        "",
        "## Who downloads",
        "",
        '<img src="segments.png" alt="Downloads by segment" width="900">',
        "",
    ]
    rows = []
    for stats in all_stats:
        downloads, ratio = stats["downloads"], weekend_ratio(stats)
        top = stats["top_clients"][0]
        rows.append(
            [
                stats["package"].name,
                share(stats["segments"]["CI"], downloads),
                share(stats["segments"]["Linux"], downloads),
                share(stats["desktop"], downloads),
                "n/a" if ratio is None else f"{ratio:.2f}",
                share(stats["installers"]["uv"], downloads),
                f"{stats['clients']:,}",
                share(int(top["n"]), downloads),
                top_shares(stats["desktop_countries"]),
            ]
        )
    lines += table(
        [
            "Package",
            "CI",
            "Linux, not CI",
            "Desktop",
            "Weekend / weekday",
            "By uv",
            "Fingerprints",
            "Top fingerprint",
            "Top desktop countries",
        ],
        rows,
    )
    lines += [
        "",
        "- **Linux, not CI** mixes servers, containers, cloud notebooks, CI that",
        "  is not flagged and Linux desktops. It can't be split further.",
        "- **Weekend / weekday**: downloads of an average Saturday or Sunday over",
        "  those of an average weekday. Scheduled jobs give 1, people at work",
        "  give a low value.",
        "- **Fingerprints**: number of distinct (country, installer, OS, Python",
        "  version, glibc version, CI flag). **Top fingerprint** is the share of",
        "  the downloads that come from the largest one. A high value on a small",
        "  package means that one machine or one pipeline makes the number (see",
        "  the [details](#details-per-package)).",
        "",
        '<img src="daily.png" alt="Downloads per day" width="900">',
        "",
        "## What they download",
        "",
    ]
    rows = []
    for stats in all_stats:
        downloads, ages = stats["downloads"], version_ages(stats)
        dated = sum(n for _, n in ages)
        latest = stats["latest"]
        median = median_age(ages)
        rows.append(
            [
                stats["package"].name,
                f"{latest} ({stats['release_dates'][latest]})",
                share(stats["versions"].get(latest, 0), downloads),
                "n/a" if median is None else months_text(median),
                share(sum(n for age, n in ages if age > 730), dated),
                share(stats["pythons"]["3.9 or older"], downloads),
                share(stats["sdist"], downloads),
            ]
        )
    lines += table(
        [
            "Package",
            "Latest release",
            "On the latest release",
            "Median release age",
            "Releases older than 2 years",
            "Python 3.9 or older",
            "From source",
        ],
        rows,
    )
    lines += [
        "",
        "- **Median release age**: half of the downloads are of a release older",
        "  than this. Old releases mean pinned environments and old Pythons, more",
        "  than new users.",
        "- **From source**: downloads of the sdist, so no wheel matched the",
        "  platform (or the user asked for a build).",
        "",
        '<img src="python.png" alt="Python versions" width="900">',
        "",
        "## PyPI and conda per month",
        "",
    ]
    if conda_months:
        month = max(conda_months)
        rows = []
        for name in names:
            channels = conda_months[month][name]
            pypi, conda = pypi_months[month][name], conda_total[month][name]
            others = {k: v for k, v in channels.items() if k != "conda-forge"}
            rows.append(
                [
                    name,
                    short(pypi),
                    short(channels["conda-forge"]),
                    ", ".join(f"{k} {short(v)}" for k, v in sorted(others.items()))
                    or "0",
                    share(conda, conda + pypi),
                ]
            )
        lines += [
            f"Downloads in {month:%B %Y}, the last month published for conda:",
            "",
        ]
        lines += table(
            [
                "Package",
                "PyPI, by an installer",
                "conda-forge",
                "Other conda channels",
                "conda share",
            ],
            rows,
        )
        lines += [
            "",
            "The conda numbers have no CI flag and no OS-level detail in common",
            "with PyPI, so the conda share compares all the conda downloads with",
            "all the PyPI downloads by an installer. `anaconda` is the defaults",
            "channel.",
            "",
            '<img src="months_conda.png" alt="conda downloads per month" width="900">',
            "",
        ]
    lines += [
        '<img src="months_pypi.png" alt="PyPI downloads per month" width="900">',
        "",
        "## Details per package",
    ]
    for stats in all_stats:
        package, downloads = stats["package"], stats["downloads"]
        lines += ["", f"### {package.name}", ""]
        if package.pypi != (package.name,):
            lines += [f"PyPI packages: {', '.join(package.pypi)}.", ""]
        lines += [
            f"- Installers: {top_shares(stats['installers'], 4)}.",
            f"- Countries, all installers: {top_shares(stats['countries'], 5)}.",
            f"- Countries, desktop: {top_shares(stats['desktop_countries'], 5)}.",
            f"- Releases: {top_shares(stats['versions'], 5)}.",
            "- Largest fingerprints:",
        ]
        for client in stats["top_clients"]:
            lines.append(
                f"  - {share(int(client['n']), downloads)}: {describe_client(client)}"
            )
    lines += [
        "",
        "## Methods and limits",
        "",
        "- **Sources.** The PyPI download logs, queried from the public ClickHouse",
        "  instance behind https://clickpy.clickhouse.com. The release dates come",
        "  from the PyPI JSON API. The conda numbers come from the monthly files",
        "  of https://github.com/anaconda/anaconda-package-data.",
        "- **A download is not an install, and not a user.** pip and uv cache the",
        "  wheels: reinstalling in a new environment on the same machine is often",
        "  not a download. A CI job or a Docker build without cache is one each",
        "  time. So the numbers measure how often the package is fetched, mostly",
        "  by machines.",
        "- **By an installer.** The user agent says which tool downloads. Mirrors",
        "  (bandersnatch, Nexus, Artifactory), browsers and scripts are dropped.",
        "  People behind a company mirror are invisible: the mirror downloads",
        "  each file once.",
        f"- **CI.** pip and uv report since {CI_FLAG_START} whether a CI environment",
        "  variable is set. poetry, pdm and old pips don't, and a Docker build",
        "  started by CI does not see the variable. So the CI share is a lower",
        "  bound, and **Linux, not CI** still has a lot of automation.",
        "- **Desktop.** Windows and macOS outside of CI. Servers and containers",
        "  almost never run these OSes, so these downloads are mostly from",
        "  laptops and workstations. It still counts each new environment of the",
        "  same person, misses WSL and Linux desktops, and misses conda users.",
        "- **Fingerprints.** The logs have no IP address or user id. Distinct",
        "  fingerprints saturate on a popular package (a few thousand",
        "  combinations exist), so this only tells how spread the downloads are,",
        "  not how many users there are.",
        "- **Countries** are those of the IP address: the US share is mostly",
        "  cloud regions.",
        "- **conda.** Monthly counts per channel, published a few days after the",
        "  end of the month, with CI and mirrors included.",
        f"- **Period.** {DAYS} days is short: a release, a holiday or one",
        "  misconfigured pipeline moves the numbers. Check the monthly plots",
        "  before reading a change as a trend.",
    ]
    (directory / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument(
        "--until",
        type=datetime.date.fromisoformat,
        default=datetime.date.today() - datetime.timedelta(days=1),
        help=f"last of the {DAYS} days, YYYY-MM-DD (default: yesterday)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "dashboard",
        help="directory to write README.md and the plots (default: dashboard/"
        " next to this script)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for package in PACKAGES:
        print(f"Fetching {package.name}")
        all_stats.append(fetch_package(package, args.until))
    months = last_months(args.until)
    print("Fetching the months")
    pypi_months = fetch_pypi_months(months)
    conda_months = fetch_conda_months(months)
    write_dashboard(all_stats, pypi_months, conda_months, args.output)
    print(f"Dashboard written to {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
