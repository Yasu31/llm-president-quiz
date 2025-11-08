#!/bin/bash

# Loop over all files ending with _tally.csv
for file in *_tally.csv; do
    # Check if any files actually match
    [ -e "$file" ] || continue

    echo "Processing $file..."
    python pie_infographic.py "$file"
done
