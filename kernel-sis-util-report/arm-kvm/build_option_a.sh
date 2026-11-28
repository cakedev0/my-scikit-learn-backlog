#!/bin/bash
# Builds linux-7.4-rc3-optA: v7.4-rc3 + option_a.patch (keep scanning until an
# idle CPU is found), from a copy of the v7.4-rc3 build tree.
set -euo pipefail
cd ~/arm-kvm
[ -d linux-7.4-rc3-optA ] || cp -a linux-7.4-rc3 linux-7.4-rc3-optA
cd linux-7.4-rc3-optA
patch -p1 --forward < ../option_a.patch || true
grep -n "best_cpu >= 0" kernel/sched/fair.c
scripts/config --set-str LOCALVERSION "-optA"
make -s olddefconfig
make -s -j"$(nproc)" Image
echo build done
