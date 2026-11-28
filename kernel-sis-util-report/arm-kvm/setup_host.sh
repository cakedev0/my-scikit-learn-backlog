#!/bin/bash
# Run once on the Graviton bare-metal host (Ubuntu 24.04 arm64): installs the
# tools and builds the guest kernels (arm64 defconfig) and the initramfs.
set -euo pipefail
cd ~/arm-kvm
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq build-essential flex bison bc \
    libssl-dev libelf-dev qemu-system-arm busybox-static device-tree-compiler cpio python3
ls -l /dev/kvm
sudo chmod 666 /dev/kvm

for v in "$@"; do
    [ -f linux-$v/arch/arm64/boot/Image ] && continue
    [ -d linux-$v ] || curl -sfL https://github.com/torvalds/linux/archive/refs/tags/v$v.tar.gz | tar xz
    make -C linux-$v -s defconfig
    make -C linux-$v -s -j"$(nproc)" Image
done

gcc -O2 -static -pthread spin_pair.c -o spin_pair
rm -rf initramfs && mkdir -p initramfs/{bin,proc,sys,dev}
cp /bin/busybox initramfs/bin/ && cp spin_pair initramfs/bin/
cp guest_init.sh initramfs/init && chmod +x initramfs/init
(cd initramfs && find . | cpio -o -H newc --quiet | gzip > ../initramfs.cpio.gz)
echo setup done
