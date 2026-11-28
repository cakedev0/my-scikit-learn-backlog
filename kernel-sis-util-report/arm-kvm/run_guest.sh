#!/bin/bash
# Usage: run_guest.sh KERNEL_VERSION TOPOLOGY OUT [kernel args]
# TOPOLOGY is "cap0,cap1,...;clusters" e.g. "530x4,1024x4;2": one capacity per
# vCPU (capacity-dmips-mhz), and the number of clusters in the cpu-map.
# Each vCPU thread is pinned to its own host core (host cores 2..N+1).
set -euo pipefail
cd ~/arm-kvm
rm -f qmp.sock
ver=$1 topo=$2 out=$3; shift 3
caps=$(python3 -c "import sys; print(' '.join(str(int(c)) for p in sys.argv[1].split(';')[0].split(',') for c in [p.split('x')[0]]*int(p.split('x')[1])))" "$topo")
ncl=${topo#*;}
n=$(wc -w <<<"$caps")
smp="$n,sockets=1,clusters=$ncl,cores=$((n / ncl)),threads=1"
qargs=(-machine virt,gic-version=3 -accel kvm -cpu host -smp "$smp" -m 4G -nographic -no-reboot)
qemu-system-aarch64 "${qargs[@]}" -machine dumpdtb=base.dtb >/dev/null 2>&1
dtc -q -I dtb -O dts base.dtb -o base.dts
python3 - "$caps" <<'PY'
import re, sys
caps = sys.argv[1].split()
s = open("base.dts").read()
seen = []
def add(m):
    i = int(m.group(1))  # QEMU names the nodes cpu@<decimal index>
    seen.append(i)
    return m.group(0) + f"\n\t\t\tcapacity-dmips-mhz = <{caps[i]}>;"
s = re.sub(r"cpu@([0-9]+) \{", add, s)
assert sorted(seen) == list(range(len(caps))), seen
open("topo.dts", "w").write(s)
PY
dtc -q -I dts -O dtb topo.dts -o topo.dtb
qemu-system-aarch64 "${qargs[@]}" -dtb topo.dtb -kernel linux-$ver/arch/arm64/boot/Image \
    -initrd initramfs.cpio.gz -append "console=ttyAMA0 loglevel=3 sched_verbose $*" \
    -qmp unix:qmp.sock,server,nowait -S </dev/null >"$out" 2>&1 &
qpid=$!
python3 - <<'PY'
import json, os, socket, time
s = socket.socket(socket.AF_UNIX)
for _ in range(100):
    try:
        s.connect("qmp.sock"); break
    except OSError:
        time.sleep(0.1)
f = s.makefile("rw")
def cmd(c):
    f.write(json.dumps({"execute": c}) + "\n"); f.flush()
    while True:
        r = json.loads(f.readline())
        if "return" in r: return r["return"]
f.readline(); cmd("qmp_capabilities")
for c in cmd("query-cpus-fast"):
    os.sched_setaffinity(c["thread-id"], {c["cpu-index"] + 2})
print("pinned", len(cmd("query-cpus-fast")), "vCPUs")
cmd("cont")
PY
wait $qpid
