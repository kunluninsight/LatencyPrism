#!/bin/bash
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
# profiler.sh
# $1: inlen
# $2: outlen
# $3: concurrency 
# $4: profiler_tp
# $5: profiler_duration
# $6: llm_pid

inlen=$1
outlen=$2
concurrency=$3
profiler_tp=$4
profiler_duration=$5
llm_pid=$6

# Profiler data will be saved in a specific directory structure
SAVE_DIR_BASE="kunlun_profile_data"
PROFILER_DATA_PATH="${SAVE_DIR_BASE}/sglang_pid${llm_pid}_tp${profiler_tp}_dur${profiler_duration}/in${inlen}_out${outlen}_conc${concurrency}"

mkdir -p "$PROFILER_DATA_PATH" || { echo "ERROR: Failed to create profiler directory $PROFILER_DATA_PATH" >&2; exit 1; }

echo "--- Kunlun Profiler Script ---"
echo "  inlen: $inlen, outlen: $outlen, concurrency: $concurrency"
echo "  TP: $profiler_tp, Duration: $profiler_duration seconds"
echo "  Saving data to: $PROFILER_DATA_PATH"
echo "------------------------------"

kunlun-profiler profile \
    --compress \
    --profile-file=command_profile.conf  --probe-config=probe_config.json --model-config=model_config.json -p $llm_pid \
    -d $profiler_duration \
    -o $PROFILER_DATA_PATH

echo "Successfully! profiler data collection finished for this combination, saved to $PROFILER_DATA_PATH"
