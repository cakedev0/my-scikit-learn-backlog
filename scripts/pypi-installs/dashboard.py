#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow"]
# ///
"""Write a dashboard of who installs a set of PyPI packages, and what.

Raw download counts are dominated by CI, containers and mirrors. This script
splits the downloads made by an installer (pip, uv, ...) by the context the
installer reports (CI flag, OS, Python version, country), and adds the release
dates from PyPI and the conda downloads published by Anaconda. The methods and
their limits are written at the end of the dashboard.

    uv run scripts/pypi-installs/dashboard.py
    uv run scripts/pypi-installs/dashboard.py --until 2026-11-19 -o some-dir

It writes one self-contained `index.html`: the numbers as JSON in
`dashboard_template.html`, which draws the tables and the plots. The
`pypi-installs-dashboard` workflow runs it every month and publishes the page
on GitHub Pages.

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

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


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
    Package("skrub"),
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
DESKTOP = ("Windows", "macOS")
TEMPLATE = pathlib.Path(__file__).with_name("dashboard_template.html")


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


def describe_client(client):
    system = {"Darwin": "macOS"}.get(client["system"], client["system"] or "?")
    details = f"{client['country'] or '?'}, {client['installer']}, {system}"
    details += f", Python {client['python']}" if client["python"] else ""
    details += f", glibc {client['glibc']}" if client["glibc"] else ""
    return details + (", CI" if client["ci"] == "true" else "")


def top(counts, count=5):
    return [[name or "?", n] for name, n in Counter(counts).most_common(count)]


# Dashboard


def package_data(stats):
    """The numbers of one package, as they go in the page."""
    package, ages = stats["package"], version_ages(stats)
    latest = stats["latest"]
    return {
        "name": package.name,
        "pypi": package.pypi,
        "all": stats["all"],
        "downloads": stats["downloads"],
        "previous": stats["previous"],
        "sdist": stats["sdist"],
        "segments": stats["segments"],
        "desktop": stats["desktop"],
        "weekend_ratio": weekend_ratio(stats),
        "installers": top(stats["installers"]),
        "uv": stats["installers"]["uv"],
        "clients": stats["clients"],
        "top_clients": [
            [describe_client(client), int(client["n"])]
            for client in stats["top_clients"]
        ],
        "countries": top(stats["countries"]),
        "desktop_countries": top(stats["desktop_countries"]),
        "versions": top(stats["versions"]),
        "latest": latest,
        "latest_day": str(stats["release_dates"][latest]),
        "latest_downloads": stats["versions"].get(latest, 0),
        "median_age_days": median_age(ages),
        "dated": sum(n for _, n in ages),
        "older_than_2_years": sum(n for age, n in ages if age > 730),
        "pythons": stats["pythons"],
        "daily": [
            {"day": str(day), **stats["daily"][day]} for day in sorted(stats["daily"])
        ],
    }


def write_dashboard(all_stats, pypi_months, conda_months, directory):
    names = [stats["package"].name for stats in all_stats]
    months = sorted(pypi_months)
    data = {
        "since": str(all_stats[0]["since"]),
        "until": str(all_stats[0]["until"]),
        "days": DAYS,
        "installers": INSTALLERS,
        "ci_flag_start": str(CI_FLAG_START),
        "packages": [package_data(stats) for stats in all_stats],
        "months": [str(month) for month in months],
        "pypi_months": {
            name: [pypi_months[month][name] for month in months] for name in names
        },
        # None for the months that Anaconda has not published yet.
        "conda_months": {
            name: [
                conda_months[month][name] if month in conda_months else None
                for month in months
            ]
            for name in names
        },
    }
    # "</" would end the script element of the page.
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text()
    if page.count("/*DATA*/null") != 1:
        raise SystemExit(f"{TEMPLATE} has no data placeholder.")
    (directory / "index.html").write_text(page.replace("/*DATA*/null", text))


def main():
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument(
        "--until",
        type=datetime.date.fromisoformat,
        # The logs of yesterday can be incomplete.
        default=datetime.date.today() - datetime.timedelta(days=2),
        help=f"last of the {DAYS} days, YYYY-MM-DD (default: 2 days ago)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("pypi-installs-dashboard"),
        help="directory to write index.html (default: pypi-installs-dashboard)",
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
    print(f"Dashboard written to {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
