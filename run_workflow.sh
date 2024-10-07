#!/bin/bash

# Navigate to the project directory
cd ~/drift_detection

# Activate the Poetry environment
source $HOME/.local/bin/poetry

# Run get_split.py and capture the output
split_info=$(poetry run python get_split.py)
if [ $? -ne 0 ]; then
    echo "There are no new splits complete so no drfit detection is performed. Exiting."
    exit 0
fi

# Parse the split_path and split_id from the output
split_path=$(echo $split_info | cut -d ' ' -f1)
split_id=$(echo $split_info | cut -d ' ' -f2)

# Run drift.py with the retrieved split_path and split_id
poetry run python drift.py --split_path $split_path --split_id $split_id

echo "Workflow completed."
