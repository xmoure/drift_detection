#!/bin/bash

# Navigate to the project directory
cd ~/drift_detection

# Run get_split.py and capture the output
split_info=$(poetry run python get_split.py)
if [ $? -ne 0 ]; then
    echo "There are no new splits complete so no drfit detection is performed. Exiting."
    exit 0
fi

# Parse the split_path and split_id from the output
split_path=$(echo $split_info | awk '{print $1}')
split_id=$(echo $split_info | awk '{print $2}')


split_path="$HOME/$split_path"

# Print split_path and split_id to verify parsing
echo "Split path: $split_path"
echo "Split ID: $split_id"

# Run concept_drift.py with the retrieved split_path and split_id
echo "Running concept drift detection on split: $split_path"
poetry run python concept_drift.py --split_path $split_path --split_id $split_id
if [ $? -ne 0 ]; then
    echo "Concept drift detection failed. Exiting."
    exit 1
fi

echo "Running drift detection on split: $split_path"
# Run drift.py with the retrieved split_path and split_id
poetry run python drift.py --split_path "$split_path" --split_id "$split_id"

echo "Workflow completed."
