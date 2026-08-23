#!/bin/bash
# run1.sh "<env assignments>" [precision] -> min kernel us + GB/s
ENVS="$1"; PREC=${2:-0}
out=$(adb shell "cd /data/local/tmp/s2dp && MNN_S2D_LOOP=8 $ENVS LD_LIBRARY_PATH=. ./run_test.out speed/FusedMathS2D 3 $PREC 68 2>/dev/null" \
      | grep -E 'kernel time = [0-9]+ +us +FusedMathS2D0' | awk '{print $4}' | sort -n | head -1)
[ -z "$out" ] && out=0
awk -v t="$out" -v e="$ENVS" 'BEGIN{ if(t>0) printf "%-46s %6d us  %5.1f GB/s\n", e, t, 21772800/(t*1e-6)/1e9; else printf "%-46s FAILED\n", e }'
