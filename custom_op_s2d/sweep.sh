#!/bin/bash
# Sweep MNN_S2D_VW x MNN_S2D_LWS on device, report min kernel time (us).
# usage: sweep.sh <precision> [extra env assignments...]
PREC=${1:-0}; shift
run() { # $1=vw $2=lws
  local env="MNN_S2D_LOOP=8"
  [ "$1" != "-" ] && env="$env MNN_S2D_VW=$1"
  [ "$2" != "0" ] && env="$env MNN_S2D_LWS=$2"
  local out
  out=$(adb shell "cd /data/local/tmp/s2dp && $env LD_LIBRARY_PATH=. ./run_test.out speed/FusedMathS2D 3 $PREC 68 2>/dev/null" \
        | grep -o 'kernel time = [0-9]* ' | awk '{print $4}' | sort -n | head -1)
  echo "$out"
}
printf "%-6s %-6s %-10s %-10s\n" VW LWS min_us GB/s
for vw in 1 2 4 8; do
  for lws in 0 64 128 256; do
    t=$(run $vw $lws)
    [ -z "$t" ] && t=0
    gbs=$(awk -v t="$t" 'BEGIN{ if(t>0) printf "%.1f", 21770977/ (t*1e-6) /1e9; else print "-" }')
    printf "%-6s %-6s %-10s %-10s\n" "$vw" "$lws" "$t" "$gbs"
  done
done
