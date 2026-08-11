#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage:
# "./print_env.sh" - prints to stdout
# "./print_env.sh > env.txt" - prints to file "env.txt"

print_env() {
echo "**git***"
if [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" == "true" ]; then
git log --decorate -n 1
echo "**git submodules***"
git submodule status --recursive
else
echo "Not inside a git repository"
fi
echo

echo "***OS Information***"
cat /etc/*-release
uname -a
echo

echo "***Docker Image***"
echo "${DOCKER_IMAGE_FULL:-(unset; not running in CI container?)}"
echo

echo "***GPU Information***"
nvidia-smi
echo

# Always-on driver-stack snapshot + Container Toolkit / CDI injection check.
#
# Background: when nvidia-container-runtime / CDI works correctly, the host's
# libcuda + libnvidia-*.so.* libs are bind-mounted into the container under
# /usr/lib/x86_64-linux-gnu/ (or similar host path). When this injection
# silently fails on a node, ldconfig falls back to the container-bundled
# /usr/local/cuda-*/compat/libcuda.so.<X> forward-compat shim. That shim is
# meant to span across major driver branches (e.g., R535 LTS userspace →
# R595 kernel), not within the same branch — so when it ends up paired with
# a kernel module whose patch level it wasn't built against, every cuda
# runtime call returns "unsupported display driver / cuda driver combination".
#
# Detection: glob for libcuda.so.* outside */compat/ and check whether any
# version matches the kernel NVRM. Empty match = injection failed. We also
# count libnvidia-*.so.* (the toolkit normally bind-mounts ~15+) as a
# secondary fingerprint.
echo "***Driver Stack***"
nvrm_ver=""
if [ -r /proc/driver/nvidia/version ]; then
    awk '/NVRM version/{sub(/^[^:]+:[[:space:]]+/,""); print "Kernel NVRM module:      " $0; exit}' /proc/driver/nvidia/version
    nvrm_ver=$(awk '/NVRM version/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]/) {print $i; exit}}' /proc/driver/nvidia/version)
else
    echo "Kernel NVRM module:      (not readable from container)"
fi
ud=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "Userspace driver (smi):  ${ud:-unknown}"
libcuda_path=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{print $NF; exit}')
echo "libcuda.so.1 (ldconfig): ${libcuda_path:-not found}"
if [ -n "$libcuda_path" ]; then
    libcuda_real=$(readlink -f "$libcuda_path" 2>/dev/null)
    [ -n "$libcuda_real" ] && [ "$libcuda_real" != "$libcuda_path" ] && echo "libcuda.so.1 (resolved): $libcuda_real"
fi

# Container Toolkit / CDI injection probe
host_libcuda_files=$(find /usr/lib /lib /opt -maxdepth 6 -name 'libcuda.so.[0-9]*' \
    -not -path '*/compat/*' 2>/dev/null | sort -u)
compat_libcuda_files=$(find /usr/local -maxdepth 6 -path '*/compat/libcuda.so.[0-9]*' 2>/dev/null | sort -u)
libnv_count=$(find /usr/lib /lib -maxdepth 4 -name 'libnvidia-*.so.[0-9]*' \
    -not -path '*/compat/*' 2>/dev/null | wc -l)

echo "Host libcuda.so.* visible (toolkit-injected, non-compat):"
if [ -z "$host_libcuda_files" ]; then
    echo "  (none)"
else
    echo "$host_libcuda_files" | sed 's/^/  /'
fi
echo "Compat libcuda.so.* (container-bundled forward-compat shim):"
if [ -z "$compat_libcuda_files" ]; then
    echo "  (none)"
else
    echo "$compat_libcuda_files" | sed 's/^/  /'
fi
echo "libnvidia-*.so.* in /usr/lib (toolkit normally injects 15+): ${libnv_count}"

# /etc/ld.so.conf.d/ ordering is the actual discriminator: when the host
# libcuda is bind-mounted (which it usually IS, even on broken nodes) but
# /etc/ld.so.conf.d/cuda-compat-*.conf ranks ahead of nvidia*.conf,
# ldconfig caches the compat libcuda first and the cuda runtime loads it
# instead of the host one — yielding the "unsupported display driver / cuda
# driver combination" error even though the host libcuda was perfectly fine.
echo "ld.so.conf.d entries (search-path order — *cuda-compat* before host nvidia.conf is the bug):"
if [ -d /etc/ld.so.conf.d ]; then
    ls -1 /etc/ld.so.conf.d 2>/dev/null | sed 's/^/  /'
else
    echo "  (no /etc/ld.so.conf.d directory)"
fi

# Verdict: what does ldconfig ACTUALLY pick for libcuda.so.1?
#   - host path + version matches kernel NVRM → OK
#   - /compat/ path                            → FAIL (regardless of whether host libcuda
#     is also present; on the broken nodes it usually IS, just shadowed)
#   - host path but version mismatch           → DEGRADED
ld_pick_real=""
if [ -n "$libcuda_path" ]; then
    ld_pick_real=$(readlink -f "$libcuda_path" 2>/dev/null || echo "$libcuda_path")
fi
ld_pick_ver=""
if [ -n "$ld_pick_real" ]; then
    ld_pick_ver=$(basename "$ld_pick_real" | sed 's/^libcuda\.so\.//')
fi

# Find the host libcuda matching the kernel (the "expected" pick) and the
# offending ld.so.conf.d entry that pushes compat ahead, for the verdict body.
expected_libcuda=""
if [ -n "$nvrm_ver" ] && [ -n "$host_libcuda_files" ]; then
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        real=$(readlink -f "$f" 2>/dev/null || echo "$f")
        fver=$(basename "$real" | sed 's/^libcuda\.so\.//')
        if [ "$fver" = "$nvrm_ver" ]; then
            expected_libcuda="$real"
            break
        fi
    done <<< "$host_libcuda_files"
fi
offending_conf=""
for _f in /etc/ld.so.conf.d/*compat*; do
    [ -e "$_f" ] || break
    offending_conf=$(basename "$_f")
    break
done

echo -n "Toolkit injection verdict: "
if [ -z "$libcuda_path" ]; then
    echo "UNKNOWN — ldconfig could not locate libcuda.so.1"
elif [[ "$ld_pick_real" == */compat/* ]]; then
    if [ -n "$expected_libcuda" ]; then
        echo "FAIL — loaded libcuda ${ld_pick_ver} != libcuda ${nvrm_ver} (expected)"
        echo "  loaded:    ${ld_pick_real} (CUDA forward-compat shim)"
        echo "  expected:  ${expected_libcuda} (host driver, matches kernel NVRM ${nvrm_ver})"
        echo "  cause:     /etc/ld.so.conf.d/${offending_conf:-cuda-compat*.conf} ranks above nvidia*.conf,"
        echo "             so ldconfig caches the compat shim instead of the host libcuda."
        echo "  fix:       rm /etc/ld.so.conf.d/${offending_conf:-cuda-compat-*.conf} && ldconfig"
        echo "             (or rename to 99-compat-*.conf so it ranks below nvidia*.conf)"
    else
        echo "FAIL — loaded libcuda ${ld_pick_ver} != kernel NVRM ${nvrm_ver:-unknown} (expected)"
        echo "  loaded:    ${ld_pick_real} (CUDA forward-compat shim)"
        echo "  expected:  host libcuda matching kernel NVRM ${nvrm_ver:-unknown} (NOT visible in container)"
        echo "  cause:     nvidia-container-runtime did not bind-mount the host driver."
        echo "  fix:       on the host node, regenerate CDI spec / restart container-runtime."
    fi
elif [ -n "$nvrm_ver" ] && [ "$ld_pick_ver" = "$nvrm_ver" ]; then
    echo "OK — loaded libcuda ${ld_pick_ver} == kernel NVRM ${nvrm_ver} (host driver in $(dirname "$ld_pick_real"))"
elif [ -z "$nvrm_ver" ]; then
    echo "UNKNOWN — kernel NVRM version not readable (loaded libcuda ${ld_pick_ver})"
else
    echo "DEGRADED — loaded libcuda ${ld_pick_ver} != kernel NVRM ${nvrm_ver}"
    echo "  loaded:    ${ld_pick_real}"
    echo "  no host libcuda matching kernel ${nvrm_ver} found in container"
fi
echo

echo "***CUDA Compute Smoke Test***"
if which nvcc &> /dev/null; then
    # Write a minimal CUDA program that tests runtime context creation
    CUDA_TEST=$(mktemp /tmp/cuda_smoke_XXXXXX.cu)
    cat > "$CUDA_TEST" << 'CUDAEOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int dev = -1, count = 0, failures = 0;
    cudaError_t err;
    err = cudaGetDeviceCount(&count);
    printf("cudaGetDeviceCount: %s (%d devices)\n", cudaGetErrorString(err), count);
    if (err != cudaSuccess || count == 0) failures++;
    err = cudaGetDevice(&dev);
    printf("cudaGetDevice:      %s (device %d)\n", cudaGetErrorString(err), dev);
    if (err != cudaSuccess) failures++;
    void *ptr = nullptr;
    err = cudaMalloc(&ptr, 256);
    printf("cudaMalloc(256):    %s (ptr=%p)\n", cudaGetErrorString(err), ptr);
    if (err != cudaSuccess) failures++;
    if (ptr) cudaFree(ptr);
    cudaStream_t stream = nullptr;
    err = cudaStreamCreate(&stream);
    printf("cudaStreamCreate:   %s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) failures++;
    if (stream) cudaStreamDestroy(stream);
    if (failures > 0) {
        printf("CUDA SMOKE TEST FAILED: %d of 4 checks failed\n", failures);
        return 1;
    }
    printf("CUDA SMOKE TEST PASSED\n");
    return 0;
}
CUDAEOF
    CUDA_BIN=$(mktemp /tmp/cuda_smoke_XXXXXX)
    if nvcc -o "$CUDA_BIN" "$CUDA_TEST" -cudart shared -Xlinker -rpath,/usr/local/cuda/lib64 2>/dev/null; then
        "$CUDA_BIN"
    else
        echo "Failed to compile CUDA smoke test"
    fi
    rm -f "$CUDA_TEST" "$CUDA_BIN"
else
    echo "nvcc not found, skipping CUDA smoke test"
fi
echo


echo "***CPU***"
lscpu
echo

echo "***CMake***"
which cmake && cmake --version
echo

echo "***g++***"
which g++ && g++ --version
echo

echo "***nvcc***"
which nvcc && nvcc --version
echo

echo "***Python***"
which python3 && python3 --version
echo

echo "***Environment Variables***"

printf '%-32s: %s\n' PATH $PATH

printf '%-32s: %s\n' LD_LIBRARY_PATH $LD_LIBRARY_PATH

printf '%-32s: %s\n' NUMBAPRO_NVVM $NUMBAPRO_NVVM

printf '%-32s: %s\n' NUMBAPRO_LIBDEVICE $NUMBAPRO_LIBDEVICE

printf '%-32s: %s\n' CONDA_PREFIX $CONDA_PREFIX

printf '%-32s: %s\n' PYTHON_PATH $PYTHON_PATH

echo


# Print conda packages if conda exists
if type "conda" &> /dev/null; then
echo '***conda packages***'
which conda && conda list
echo
# Print pip packages using system python
elif type "python3" &> /dev/null; then
echo "conda not found"
echo "***pip packages***"
which pip && python3 -m pip list
echo
else
echo "conda not found"
echo "pip not found"
fi
}

echo "<details><summary>Click here to see environment details</summary><pre>"
echo "     "
print_env | while read -r line; do
    echo "     $line"
done
echo "</pre></details>"

# When the smoke test fails, emit a tight copy-paste-ready incident ticket.
# Skips fields already printed above (full Driver Stack, smoke output) — the
# ticket only carries the load-bearing info: PROBLEM (loaded != expected),
# pod / build URL / GPU UUID-serial-bus, and the host-side fix command. The
# verbose Driver Stack section above is referenced as EVIDENCE.
print_smoke_diagnostics() {
    local reason="$1"
    # $2 (smoke stdout) is intentionally unused in the trimmed ticket; the
    # full smoke output is already printed above in ***CUDA Compute Smoke Test***
    local ts pod node build_url gpu_line offending loaded_path loaded_ver expected_path nvrm_ver_local

    ts=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
    pod="${POD_NAME:-${HOSTNAME:-$(hostname 2>/dev/null || echo unknown)}}"
    node="${K8S_NODE_NAME:-unknown}"
    build_url="${BUILD_URL:-N/A}"

    # GPU one-liner: name uuid=... serial=... bus=...
    gpu_line=$(nvidia-smi --query-gpu=name,uuid,serial,pci.bus_id --format=csv,noheader 2>/dev/null \
                | head -1 | awk -F', ' '{printf "%s  uuid=%s  serial=%s  bus=%s", $1, $2, $3, $4}')

    # Re-derive the loaded vs expected libcuda paths and the offending conf
    # file. Best-effort — if anything is missing the field renders as "?".
    nvrm_ver_local=$(awk '/NVRM version/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]/) {print $i; exit}}' /proc/driver/nvidia/version 2>/dev/null)
    loaded_path=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{print $NF; exit}')
    [ -n "$loaded_path" ] && loaded_path=$(readlink -f "$loaded_path" 2>/dev/null || echo "$loaded_path")
    loaded_ver=$(basename "${loaded_path:-libcuda.so.0}" | sed 's/^libcuda\.so\.//')
    expected_path=$(find /usr/lib /lib -maxdepth 4 -name "libcuda.so.${nvrm_ver_local}" -not -path '*/compat/*' 2>/dev/null | head -1)
    offending=""
    for _f in /etc/ld.so.conf.d/*compat*; do
        [ -e "$_f" ] || break
        offending=$(basename "$_f")
        break
    done

    echo
    echo "================== INCIDENT TICKET (paste into bug) =================="
    echo "PROBLEM:  cvcuda CI smoke test failed."
    if [[ "$loaded_path" == */compat/* ]] && [ -n "$expected_path" ]; then
        echo "          Loaded libcuda ${loaded_ver} != ${nvrm_ver_local} (expected — matches kernel NVRM)."
        echo "          Cause: /etc/ld.so.conf.d/${offending:-cuda-compat*.conf} shadows nvidia.conf."
    else
        echo "          ${reason}"
        echo "          Loaded libcuda: ${loaded_path:-?}  Kernel NVRM: ${nvrm_ver_local:-?}"
    fi
    echo
    echo "WHEN:     ${ts}"
    echo "POD:      ${pod}"
    echo "NODE:     ${node}"
    echo "BUILD:    ${build_url}"
    echo "GPU:      ${gpu_line:-?}"
    echo
    if [[ "$loaded_path" == */compat/* ]] && [ -n "$offending" ]; then
        echo "FIX:      On the broken node:"
        echo "            rm /etc/ld.so.conf.d/${offending} && ldconfig"
        echo "          (or rename to 99-${offending#*-} to deprioritize without removing)"
        echo "          Verify:  docker run --rm --gpus all nvidia/cuda:13.3.0-base-ubuntu26.04 \\"
        echo "                     bash -c 'ldconfig -p | grep libcuda.so.1'"
        echo "          should print /usr/lib/x86_64-linux-gnu/libcuda.so.1 (not /compat/)."
    else
        echo "FIX:      Inspect the Driver Stack section above for the loaded vs expected"
        echo "          libcuda paths and the ld.so.conf.d listing."
    fi
    echo
    echo "EVIDENCE: ***Driver Stack*** section above — FAIL verdict + full ld.so.conf.d listing."
    echo "======================================================================"
    echo
}

# Re-run the CUDA smoke test outside the function to capture exit code
# (the one inside print_env runs in a subshell via pipe, so its exit code is lost)
if which nvcc &> /dev/null; then
    CUDA_TEST=$(mktemp /tmp/cuda_check_XXXXXX.cu)
    # Same checks as the verbose smoke test above, but emits one labeled line
    # per cuda call so the gating-stage stdout is itself ticket-worthy.
    cat > "$CUDA_TEST" << 'EOF2'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int dev = -1, count = 0, failures = 0;
    cudaError_t err;
    err = cudaGetDeviceCount(&count);
    printf("cudaGetDeviceCount: %s (%d devices)\n", cudaGetErrorString(err), count);
    if (err != cudaSuccess || count == 0) failures++;
    err = cudaGetDevice(&dev);
    printf("cudaGetDevice:      %s (device %d)\n", cudaGetErrorString(err), dev);
    if (err != cudaSuccess) failures++;
    void *ptr = 0;
    err = cudaMalloc(&ptr, 256);
    printf("cudaMalloc(256):    %s (ptr=%p)\n", cudaGetErrorString(err), ptr);
    if (err != cudaSuccess) failures++;
    if (ptr) cudaFree(ptr);
    cudaStream_t stream = 0;
    err = cudaStreamCreate(&stream);
    printf("cudaStreamCreate:   %s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) failures++;
    if (stream) cudaStreamDestroy(stream);
    return failures > 0 ? 1 : 0;
}
EOF2
    CUDA_BIN=$(mktemp /tmp/cuda_check_XXXXXX)
    NVCC_ERR=$(mktemp /tmp/cuda_nvcc_err_XXXXXX)
    if ! nvcc -o "$CUDA_BIN" "$CUDA_TEST" -cudart shared -Xlinker -rpath,/usr/local/cuda/lib64 2>"$NVCC_ERR"; then
        echo "ERROR: Failed to compile CUDA smoke test ($CUDA_TEST -> $CUDA_BIN)"
        nvcc_stderr=$(cat "$NVCC_ERR")
        echo "nvcc stderr:"
        echo "$nvcc_stderr"
        print_smoke_diagnostics "nvcc failed to compile smoke test" "$nvcc_stderr"
        rm -f "$CUDA_TEST" "$CUDA_BIN" "$NVCC_ERR"
        exit 1
    fi
    rm -f "$NVCC_ERR"
    smoke_out=$("$CUDA_BIN" 2>&1)
    smoke_rc=$?
    if [ $smoke_rc -ne 0 ]; then
        echo "$smoke_out"
        # Last-resort retry: try the libcuda-ordering self-heal script and
        # re-run the smoke test once. The script only acts when the host
        # libcuda matches the kernel NVRM (i.e. the broken-node ld.so.conf.d
        # ordering pattern); legitimate forward-compat use cases on older
        # host drivers leave it untouched. If the script applies a fix, it
        # exits 0 and we retry; otherwise we go straight to emitting the
        # incident ticket.
        SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        FIX_SCRIPT="${SCRIPT_DIR}/ci/fix_libcuda_ldconfig.sh"
        if [ -x "$FIX_SCRIPT" ]; then
            echo
            echo "*** Smoke test failed — attempting self-heal via ${FIX_SCRIPT##*/} ***"
            if "$FIX_SCRIPT"; then
                echo "*** Self-heal applied, retrying smoke test ***"
                smoke_out=$("$CUDA_BIN" 2>&1)
                smoke_rc=$?
                if [ $smoke_rc -eq 0 ]; then
                    echo "$smoke_out"
                    echo "*** Smoke test PASSED after libcuda ldconfig self-heal ***"
                    rm -f "$CUDA_TEST" "$CUDA_BIN"
                    return 0 2>/dev/null || exit 0
                fi
                echo "$smoke_out"
                echo "*** Smoke test still failing after self-heal — giving up ***"
            else
                echo "*** Self-heal not applicable on this node — giving up ***"
            fi
        fi
        print_smoke_diagnostics "cuda runtime call failed (driver/userspace mismatch likely)" "$smoke_out"
        echo "ERROR: CUDA compute smoke test failed on this node. Check GPU/driver compatibility."
        rm -f "$CUDA_TEST" "$CUDA_BIN"
        exit 1
    fi
    rm -f "$CUDA_TEST" "$CUDA_BIN"
fi
