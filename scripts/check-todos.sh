set -e

todos_found=0
for f in $(find tex -type f -name '*.tex' -not -path 'tex/ltex/*'); do
  todos_found_it=0
  if grep -qi '^[[:space:]]*%[[:space:]]*todo.*$' "$f"; then
    match=$(grep -i '^.*%[[:space:]]*todo.*$' "$f" | sed 's/^.*%[[:space:]]*//')
    todos_found_it=1
    if [ $todos_found -eq 0 ]; then
      echo "There are todos in your latex files:"
      todos_found=1
    fi
    if [ $todos_found_it -eq 1 ]; then
      echo "$match" | while IFS= read -r line; do
        echo -e "  $f:\t$line"
      done
    fi
  fi
done

pylint --disable=all --enable=W0511 python scripts ./notebooks/as_scripts test/*.py

if [ $todos_found -eq 1 ]; then
  exit 1
fi