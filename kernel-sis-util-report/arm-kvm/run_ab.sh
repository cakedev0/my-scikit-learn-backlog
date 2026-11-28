#!/bin/bash
# A/B: v7.4-rc3 vs v7.4-rc3-optA, alternating boots, REPS times per topology.
set -u
cd ~/arm-kvm && mkdir -p results
declare -A T=(
    [rk3588]="530x4,1024x4;2"
    [bigfirst]="1024x4,530x4;2"
    [phone]="400x4,850x3,1024x1;1"
    [desktop]="1024x8,701x16;6"
)
for rep in $(seq "${REPS:-2}"); do
    for name in desktop rk3588 bigfirst phone; do
        for ver in 7.4-rc3 7.4-rc3-optA; do
            echo "$(date +%T) rep $rep $name $ver"
            ./run_guest.sh "$ver" "${T[$name]}" "results/ab-$name-$ver-rep$rep.log" rounds=3 runs=10
        done
    done
done
echo ALL DONE
