#!/bin/bash
# All the topologies, on v7.4-rc3 and v7.1. One log per (topology, kernel) in results/.
set -u
cd ~/arm-kvm && mkdir -p results
declare -A T=(
    [rk3588]="530x4,1024x4;2"          # little first, like the RK3588 DT
    [bigfirst]="1024x4,530x4;2"        # same capacities, x86-like order
    [phone]="400x4,850x3,1024x1;1"     # 4 little + 3 mid + 1 prime, one DynamIQ cluster
    [desktop]="1024x8,701x16;6"        # like the Core Ultra 9 285: 8 P + 16 E
)
for name in desktop rk3588 bigfirst phone; do
    for ver in 7.4-rc3 7.1; do
        echo "$(date +%T) $name $ver"
        ./run_guest.sh "$ver" "${T[$name]}" "results/$name-$ver${SUFFIX:-}.log" rounds=${ROUNDS:-2} runs=${RUNS:-4}
    done
done
echo ALL DONE
