#!/bin/busybox sh
# /init of the guest: prints the topology, then alternates SIS_UTIL and
# NO_SIS_UTIL, ROUNDS times, with RUNS spin_pair runs each time, then powers off.
/bin/busybox --install -s /bin
export PATH=/bin
mount -t proc proc /proc; mount -t sysfs sys /sys; mount -t devtmpfs dev /dev
mount -t debugfs none /sys/kernel/debug
ROUNDS=2 RUNS=4 GAP=6000
for a in $(cat /proc/cmdline); do case $a in rounds=*|runs=*|gap=*) eval "$(echo ${a%%=*} | tr a-z A-Z)=${a#*=}";; esac; done
echo "kernel $(uname -r), $(nproc) CPUs, HZ config: $(zcat /proc/config.gz 2>/dev/null | grep '^CONFIG_HZ=' || echo n/a)"
for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "${c##*/} capacity $(cat $c/cpu_capacity)"; done
for d in /sys/kernel/debug/sched/domains/cpu*/domain*; do echo "$d $(cat $d/name) $(cat $d/flags)"; done
echo "energy aware: $(cat /proc/sys/kernel/sched_energy_aware 2>/dev/null || echo n/a)"
r=1
while [ $r -le $ROUNDS ]; do
    for f in SIS_UTIL NO_SIS_UTIL; do
        echo $f > /sys/kernel/debug/sched/features
        echo "### round $r, $f"
        i=0; while [ $i -lt $RUNS ]; do spin_pair 150 100 $GAP; i=$((i+1)); done
        SPIN_PIN=1 spin_pair 150 100 $GAP | sed 's/^/pinned: /'
    done
    r=$((r+1))
done
echo DONE
poweroff -f
