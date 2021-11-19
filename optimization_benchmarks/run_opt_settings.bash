#!/bin/bash

# https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euo pipefail

printf "Running snakemake...\n"
snakemake \
    -s Snakefile_opt_settings \
    -j 36 \
    --latency-wait 60 \
    --keep-going
printf "Run of snakemake complete.\n"
