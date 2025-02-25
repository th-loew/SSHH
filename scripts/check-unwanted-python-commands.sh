#!/bin/bash

patterns="^\s*print\(|np.ndarray|Union|Optional|List|Tuple|Dict|NDArray"

# Search recursively for the 'print(' function at the beginning of a line, with optional leading whitespaces
grep -rnHP --include="*.py" $patterns ./python ./scripts | grep -v "#.*check: ignore" | grep -v "import "

# Exit status, 0 if found, 1 otherwise
if [ $? -eq 0 ]; then
  echo "Found unwanted statements in the code."
  exit 1
else
  echo "No unwanted statements found in the code."
fi
