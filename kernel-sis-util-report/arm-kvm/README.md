# SIS_UTIL stalls on emulated asymmetric-capacity topologies (arm64 KVM)

Can the stalls of `../report.txt` happen on other CPU numberings than hybrid x86
(little cores first, as on most Arm SoCs)? There is no asymmetric Arm hardware
here, so the test runs a KVM guest on a Graviton3 bare-metal host (`c7g.metal`)
whose device tree gives each vCPU a `capacity-dmips-mhz`. All host cores are
identical: only the guest scheduler sees different capacities. Each vCPU is
pinned to its own host core. Guest kernels: mainline v7.4-rc3 and v7.1, arm64
defconfig (HZ=250), no EAS. Workload: `spin_pair 150 100 6000`, one thread
per vCPU; on arm64 the spin hint is `isb` (300000 iterations = 4.2 ms here).

- `setup_host.sh 7.4-rc3 7.1`: tools, kernels and initramfs on the host.
- `run_guest.sh VERSION "cap x count,...;clusters" LOG`: one guest boot.
- `run_all.sh`: the four topologies on both kernels (`ROUNDS`, `RUNS`).
- `guest_init.sh`: the guest's `/init`, alternates SIS_UTIL and NO_SIS_UTIL.
- `summarize.py results/*-long.log` or `results/ab-*.log`: the tables below.
- `option_a.patch`, `build_option_a.sh`, `run_ab.sh`: the Option A test.

Topologies: `rk3588` CPUs 0-3 = 530, 4-7 = 1024, 2 clusters (little first);
`bigfirst` same capacities, big first (x86-like order); `phone` 4 x 400,
3 x 850, 1 x 1024, one cluster; `desktop` 8 x 1024 + 16 x 701, like the Core
Ultra 9 285.

3 rounds of 10 runs per setting, alternated in one boot. A stalled run has
>= 10 episodes (many short stalls), not the rare single episode of a whole call:

```
topology   kernel   feature      runs stalled  regions > 1 ms (min..max, total)  episodes (max)
bigfirst   7.1      NO_SIS_UTIL    30       0  0..100, 107               1
bigfirst   7.1      SIS_UTIL       30       0  0..1, 3                 1
bigfirst   7.4-rc3  NO_SIS_UTIL    30       0  0..1, 2                 1
bigfirst   7.4-rc3  SIS_UTIL       30       6  0..469, 2274              55
desktop    7.1      NO_SIS_UTIL    30       0  0..6, 9                 2
desktop    7.1      SIS_UTIL       30       0  0..4, 8                 1
desktop    7.4-rc3  NO_SIS_UTIL    30       0  0..4, 6                 1
desktop    7.4-rc3  SIS_UTIL       30       2  0..4220, 7739              110
phone      7.1      NO_SIS_UTIL    30       0  0..4, 5                 1
phone      7.1      SIS_UTIL       30       0  0..3, 3                 1
phone      7.4-rc3  NO_SIS_UTIL    30       0  0..0, 0                 0
phone      7.4-rc3  SIS_UTIL       30       1  0..183, 201               25
rk3588     7.1      NO_SIS_UTIL    30       0  0..54, 58                1
rk3588     7.1      SIS_UTIL       30       0  0..1, 2                 1
rk3588     7.4-rc3  NO_SIS_UTIL    30       0  0..4, 8                 1
rk3588     7.4-rc3  SIS_UTIL       30       2  0..344, 686               37
```

Stalled runs only happen with v7.4-rc3 and SIS_UTIL, on all four topologies
(11 of 120 runs), and in none of the 360 other runs. So the little-first
numbering doesn't protect from them. They are much rarer than on the real
desktop (all 16 SIS_UTIL runs stalled there vs 2 of 30 here for the
`desktop` topology), probably because the "little" vCPUs aren't actually
slower. `bigfirst` vs `rk3588` (6 vs 2 of 30) is too few runs to say the
order matters. The logs without `-long` are a first
series of 2 rounds of 4 runs, same conclusion.

## Option A: keep scanning until an idle CPU is found

`option_a.patch` on v7.4-rc3 (`build_option_a.sh`): the nr_idle_scan limit
of select_idle_capacity() only stops the scan once an idle CPU has been
found, so it returns -1 only when no CPU of the span is idle. A/B with
`run_ab.sh`: per topology, boots of v7.4-rc3 and v7.4-rc3-optA alternated
twice, 3 rounds of 10 runs per setting in each boot:

```
topology   kernel   feature      runs stalled  regions > 1 ms (min..max, total)  episodes (max)
bigfirst   7.4-rc3  NO_SIS_UTIL    60       0  0..3, 7                 1
bigfirst   7.4-rc3  SIS_UTIL       60       9  0..441, 3052              45
bigfirst   7.4-rc3-optA NO_SIS_UTIL    60       0  0..1, 8                 1
bigfirst   7.4-rc3-optA SIS_UTIL       60       0  0..40, 43                2
desktop    7.4-rc3  NO_SIS_UTIL    60       0  0..3, 4                 1
desktop    7.4-rc3  SIS_UTIL       60       5  0..5200, 20434             123
desktop    7.4-rc3-optA NO_SIS_UTIL    60       0  0..4, 12                1
desktop    7.4-rc3-optA SIS_UTIL       60       0  0..0, 0                 0
phone      7.4-rc3  NO_SIS_UTIL    60       0  0..0, 0                 0
phone      7.4-rc3  SIS_UTIL       60       9  0..244, 1782              41
phone      7.4-rc3-optA NO_SIS_UTIL    60       0  0..0, 0                 0
phone      7.4-rc3-optA SIS_UTIL       60       0  0..0, 0                 0
rk3588     7.4-rc3  NO_SIS_UTIL    60       0  0..2, 5                 2
rk3588     7.4-rc3  SIS_UTIL       60       9  0..442, 2635              46
rk3588     7.4-rc3-optA NO_SIS_UTIL    60       0  0..8, 25                1
rk3588     7.4-rc3-optA SIS_UTIL       60       0  0..1, 2                 1
```

With SIS_UTIL, 32 of 240 runs stall on v7.4-rc3 and none of 240 with the
patch. In the runs without stalls the median call time is the same with and
without the patch (0.21 to 0.36 ms depending on the topology). This workload
doesn't measure what SIS_UTIL protects (the scan cost on a large, busy
domain with no idle CPU), so it says nothing about that cost.
