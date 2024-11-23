#!/bin/bash

if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Should've installed requirements first. Installing gdown..."
    pip install gdown
fi

FILE_URL="https://drive.google.com/uc?id=1W_QE5pFskApv1cNv8IwpSNW1xzvA6xia"
OUTPUT_FILE="lrw_resnet18_mstcn_video.pth"

if [ -f "$OUTPUT_FILE" ]; then
    echo "File '$OUTPUT_FILE' already exists. Skipping download."
else
    echo "Downloading '$OUTPUT_FILE' with gdown..."
    gdown "$FILE_URL" -O "$OUTPUT_FILE"
    echo "SUCCESS: $OUTPUT_FILE downloaded."
fi
