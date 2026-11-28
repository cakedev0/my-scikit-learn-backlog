"""Summarize the guest logs: per (topology, kernel, sched feature), how many
runs have >= 10 episodes ("stalled runs": many short stalls, the SIS_UTIL kind,
not the rare single episode of a whole call), and the regions > 1 ms.
Usage: python summarize.py results/*-long.log  (or results/ab-*.log)"""
import re
import sys
from collections import defaultdict

rows = defaultdict(list)
for path in sys.argv[1:]:
    name = path.rsplit("/", 1)[-1].removesuffix(".log").removesuffix("-long")
    name = re.sub(r"-rep\d+$", "", name.removeprefix("ab-"))
    topo, ver = name.split("-", 1)
    feat = None
    for line in open(path, errors="replace"):
        if m := re.match(r"### round \d+, (\S+)", line):
            feat = m.group(1)
        elif line.startswith("threads=") and feat:
            slow = int(re.search(r"regions > 1 ms: (\d+)/", line).group(1))
            ep = int(re.search(r"episodes: (\d+)", line).group(1))
            rows[topo, ver, feat].append((slow, ep))

print(f"{'topology':10} {'kernel':8} {'feature':12} {'runs':>4} {'stalled':>7}  regions > 1 ms (min..max, total)  episodes (max)")
for (topo, ver, feat), v in sorted(rows.items()):
    slow = [s for s, _ in v]
    stalled = sum(e >= 10 for _, e in v)
    print(f"{topo:10} {ver:8} {feat:12} {len(v):4} {stalled:7}  "
          f"{min(slow)}..{max(slow)}, {sum(slow):<6}            {max(e for _, e in v)}")
