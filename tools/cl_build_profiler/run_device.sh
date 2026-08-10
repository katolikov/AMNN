#!/usr/bin/env bash
#
#  run_device.sh
#  MNN
#
#  Builds the clBuildProgram profiler for Android, pushes it to a device, runs it and
#  pulls the results back. Everything after `--` is forwarded to the profiler.
#
#  Copyright © 2018, Alibaba Group Holding Limited
#

set -euo pipefail

readonly SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MNN_ROOT="$(cd -P "${SCRIPT_DIR}/../.." && pwd)"

DEVICE_SERIAL=""
ABI="arm64-v8a"
API_LEVEL="24"
NDK_PATH="${ANDROID_NDK:-${ANDROID_NDK_HOME:-${NDK_ROOT:-}}}"
BUILD_DIR="${SCRIPT_DIR}/build-android"
REMOTE_DIR="/data/local/tmp/cl_build_profiler"
OUTPUT_DIR=""
DO_BUILD=1
DO_PUSH=1
PROFILER_ARGS=()

usage() {
    cat <<'EOF'
usage: run_device.sh [-s <serial>] [options] [-- <profiler arguments>]

  -s, --serial <serial>   target device, required when more than one is connected
      --abi <abi>         Android ABI to build (default arm64-v8a)
      --api <level>       Android API level to build against (default 24)
      --ndk <path>        NDK location (default $ANDROID_NDK / $ANDROID_NDK_HOME / $NDK_ROOT)
      --build-dir <path>  host build directory (default <script dir>/build-android)
      --remote-dir <path> device directory (default /data/local/tmp/cl_build_profiler)
  -o, --output <path>     host directory for the pulled results (default ./cl_profile_<time>)
      --no-build          reuse the binary already in the build directory
      --no-push           reuse the binary already on the device
  -h, --help              show this message

Everything after -- goes to the profiler, for example:

  ./run_device.sh -s R5CY71BJJ9D -- --programs 'conv_2d*' --repeat 5 --jobs 4

The profiler exit code is propagated: 0 ok, 1 fatal, 2 a program failed to build,
3 a verification check failed.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

# The profiler arguments are re-parsed by the shell on the device, so each one is
# single quoted here; without it a pattern such as 'conv*' would glob remotely.
quote_for_shell() {
    local quoted=""
    local argument
    for argument in "$@"; do
        quoted+=" '${argument//\'/\'\\\'\'}'"
    done
    printf '%s' "${quoted}"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -s|--serial)     [[ $# -ge 2 ]] || die "$1 needs a value"; DEVICE_SERIAL="$2"; shift 2 ;;
            --abi)           [[ $# -ge 2 ]] || die "$1 needs a value"; ABI="$2"; shift 2 ;;
            --api)           [[ $# -ge 2 ]] || die "$1 needs a value"; API_LEVEL="$2"; shift 2 ;;
            --ndk)           [[ $# -ge 2 ]] || die "$1 needs a value"; NDK_PATH="$2"; shift 2 ;;
            --build-dir)     [[ $# -ge 2 ]] || die "$1 needs a value"; BUILD_DIR="$2"; shift 2 ;;
            --remote-dir)    [[ $# -ge 2 ]] || die "$1 needs a value"; REMOTE_DIR="$2"; shift 2 ;;
            -o|--output)     [[ $# -ge 2 ]] || die "$1 needs a value"; OUTPUT_DIR="$2"; shift 2 ;;
            --no-build)      DO_BUILD=0; shift ;;
            --no-push)       DO_PUSH=0; shift ;;
            -h|--help)       usage; exit 0 ;;
            --)              shift; PROFILER_ARGS=("$@"); break ;;
            *)               die "unknown argument: $1" ;;
        esac
    done
}

# adb with the selected serial, so nothing can act on the wrong device.
adb_run() {
    if [[ -n "${DEVICE_SERIAL}" ]]; then
        adb -s "${DEVICE_SERIAL}" "$@"
    else
        adb "$@"
    fi
}

resolve_device() {
    command -v adb >/dev/null 2>&1 || die "adb is not on PATH"

    local devices
    devices="$(adb devices | awk 'NR > 1 && $2 == "device" { print $1 }')"
    [[ -n "${devices}" ]] || die "no device is connected (adb devices shows none in state 'device')"

    if [[ -z "${DEVICE_SERIAL}" ]]; then
        local count
        count="$(echo "${devices}" | wc -l | tr -d ' ')"
        [[ "${count}" == "1" ]] || die "$(printf 'several devices connected, pass -s <serial>:\n%s' "${devices}")"
        DEVICE_SERIAL="${devices}"
    elif ! echo "${devices}" | grep -qx "${DEVICE_SERIAL}"; then
        die "$(printf 'device %s is not connected. Available:\n%s' "${DEVICE_SERIAL}" "${devices}")"
    fi
    echo "device: ${DEVICE_SERIAL} ($(adb_run shell getprop ro.product.model | tr -d '\r'))"
}

resolve_ndk() {
    if [[ -z "${NDK_PATH}" ]]; then
        # Version sort is preferred but not available in every sort implementation.
        local sorter=(sort)
        printf '1\n' | sort -V >/dev/null 2>&1 && sorter=(sort -V)

        local candidates=("${HOME}/Library/Android/sdk/ndk" "${HOME}/Android/Sdk/ndk" "/opt/android-ndk")
        local root
        for root in "${candidates[@]}"; do
            [[ -d "${root}" ]] || continue
            NDK_PATH="$(find "${root}" -maxdepth 1 -mindepth 1 -type d | "${sorter[@]}" | tail -1)"
            [[ -n "${NDK_PATH}" ]] && break
        done
    fi
    [[ -n "${NDK_PATH}" ]] || die "no NDK found, pass --ndk <path> or set ANDROID_NDK"
    local toolchain="${NDK_PATH}/build/cmake/android.toolchain.cmake"
    [[ -f "${toolchain}" ]] || die "${toolchain} does not exist"
    echo "ndk:    ${NDK_PATH}"
}

build_profiler() {
    echo "==> building ${ABI} (api ${API_LEVEL})"
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${NDK_PATH}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="${ABI}" \
        -DANDROID_PLATFORM="android-${API_LEVEL}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_ROOT="${MNN_ROOT}" >/dev/null
    cmake --build "${BUILD_DIR}" -j "$(getconf _NPROCESSORS_ONLN)"
}

push_profiler() {
    echo "==> pushing to ${REMOTE_DIR}"
    adb_run shell "mkdir -p ${REMOTE_DIR}"
    adb_run push "${BUILD_DIR}/cl_build_profiler" "${REMOTE_DIR}/cl_build_profiler" >/dev/null
    adb_run shell "chmod 755 ${REMOTE_DIR}/cl_build_profiler"
}

# Recorded next to the results: build times mean nothing without knowing which SoC,
# which driver and at which clocks they were taken.
collect_device_facts() {
    local target="$1"
    {
        echo "serial          : ${DEVICE_SERIAL}"
        for property in ro.product.model ro.product.name ro.board.platform ro.hardware \
                        ro.build.version.release ro.build.version.sdk ro.build.fingerprint; do
            printf '%-16s: %s\n' "${property}" "$(adb_run shell getprop "${property}" | tr -d '\r')"
        done
        echo
        echo "cpu governors and current frequencies"
        adb_run shell 'for policy in /sys/devices/system/cpu/cpufreq/policy*; do
            [ -d "$policy" ] || continue
            echo "  $(basename $policy): governor=$(cat $policy/scaling_governor 2>/dev/null) cur=$(cat $policy/scaling_cur_freq 2>/dev/null) max=$(cat $policy/scaling_max_freq 2>/dev/null)"
        done' 2>/dev/null | tr -d '\r'
        echo
        echo "thermal zones above 40C"
        adb_run shell 'for zone in /sys/class/thermal/thermal_zone*; do
            [ -r "$zone/temp" ] || continue
            temp=$(cat $zone/temp 2>/dev/null)
            case "$temp" in ""|*[!0-9]*) continue;; esac
            [ "$temp" -gt 40000 ] && echo "  $(cat $zone/type 2>/dev/null): $((temp / 1000))C"
        done' 2>/dev/null | tr -d '\r'
    } > "${target}" 2>&1 || true
}

main() {
    parse_args "$@"
    resolve_device

    if [[ -z "${OUTPUT_DIR}" ]]; then
        OUTPUT_DIR="${PWD}/cl_profile_$(date +%Y%m%d_%H%M%S)"
    fi

    if [[ "${DO_BUILD}" == "1" ]]; then
        resolve_ndk
        build_profiler
    else
        [[ -x "${BUILD_DIR}/cl_build_profiler" ]] || die "${BUILD_DIR}/cl_build_profiler does not exist"
    fi

    if [[ "${DO_PUSH}" == "1" ]]; then
        push_profiler
    fi

    mkdir -p "${OUTPUT_DIR}"
    collect_device_facts "${OUTPUT_DIR}/device_info.txt"

    echo "==> running on device"
    local remote_args=""
    if [[ ${#PROFILER_ARGS[@]} -gt 0 ]]; then
        remote_args="$(quote_for_shell "${PROFILER_ARGS[@]}")"
    fi

    local status
    set +e
    adb_run shell "cd ${REMOTE_DIR} && ./cl_build_profiler --csv ${REMOTE_DIR}/samples.csv --json ${REMOTE_DIR}/summary.json${remote_args} 2>&1; echo EXIT_STATUS=\$?" \
        | tr -d '\r' | tee "${OUTPUT_DIR}/profile.log"
    set -e
    status="$(awk -F= '/^EXIT_STATUS=/ { print $2 }' "${OUTPUT_DIR}/profile.log" | tail -1)"
    [[ "${status}" =~ ^[0-9]+$ ]] || status=1
    sed -i.bak '/^EXIT_STATUS=/d' "${OUTPUT_DIR}/profile.log" && rm -f "${OUTPUT_DIR}/profile.log.bak"

    adb_run pull "${REMOTE_DIR}/samples.csv" "${OUTPUT_DIR}/samples.csv" >/dev/null 2>&1 || true
    adb_run pull "${REMOTE_DIR}/summary.json" "${OUTPUT_DIR}/summary.json" >/dev/null 2>&1 || true

    echo
    echo "results in ${OUTPUT_DIR}"
    ls -1 "${OUTPUT_DIR}"
    exit "${status}"
}

main "$@"
