#!/bin/sh

OUTPUT_DIR="$1"

# For each file in the output directory, calculate statistical significance
for file in ${OUTPUT_DIR}/*-scores.tsv; do
    testname=$(basename $file)
    testname=${testname%%-scores.tsv}
    python ../scripts/statistical_significance.py "${file}" > "${OUTPUT_DIR}/${testname}-significance.txt"
done
