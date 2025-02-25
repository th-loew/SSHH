#!/usr/bin/env bash

set -e

# Run coverage
# Read the coverage blacklist
# Check if coverage.blacklist exists and read it, else initialize blacklist as empty
if [ -f .coverage-blacklist ]; then
    mapfile -t blacklist < .coverage-blacklist
else
    blacklist=()
fi
# Function to check if a file is in the blacklist
is_blacklisted() {
    local file=$1
    for blacklisted in "${blacklist[@]}"; do
        if [[ $file == $blacklisted ]]; then
            return 0
        fi
    done
    return 1
}
COVERAGE_MIN=90
coverage_ok=1

python -m coverage report --include="python/*" --omit "python/plotting*.py"
python -m coverage html --include="python/*" --omit "python/plotting*.py"

# Check coverage for all files in the python directory
for file in python/*.py; do
    # Check if the file is blacklisted
    if is_blacklisted "$file"; then
        continue
    fi
    # Generate coverage report for the file
    if FILE_COVERAGE=$(python -m coverage report -m | grep "$file"); then
      # Extract the coverage percentage
      COVERAGE_PERCENT=$(echo $FILE_COVERAGE | awk -F'%' '{print $1}' | awk '{print $NF}')
      # Check if coverage is at least 90%
      if [ "$COVERAGE_PERCENT" -lt "$COVERAGE_MIN" ]; then
          echo "Coverage for $file is below $COVERAGE_MIN% ($COVERAGE_PERCENT%)"
          coverage_ok=0
      fi
    else
      echo "$file is not tested"
      coverage_ok=0
    fi
done
# exit if coverage not ok
if [ $coverage_ok -eq 0 ]; then
    exit 1
else
    echo "Test coverage is OK"
fi
