#!/bin/bash

MAIN_DIR="."
INPUT_DIR="${MAIN_DIR}/sample_imgs/input/"
OUTPUT_DIR="${MAIN_DIR}/sample_imgs/output/"
PRETRAINED="${MAIN_DIR}/pretrained/DGSS_loss=bce-rgb_date=23Feb2021.tar"

python3 analysis_single_nC.py \
--pretrained $PRETRAINED \
--input_dir $INPUT_DIR \
--output_dir $OUTPUT_DIR \
--nC 100 \
