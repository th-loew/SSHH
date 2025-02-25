set dotenv-load

alias chmod := git-scripts-executable
alias build-plots := build
alias clean-coverage := clean
alias test-python := test

_default:
    @just --list

# Run all tests, builds and checks
all build_all_plots='0':
    @just test
    @just build {{build_all_plots}}
    @just check

# Build plots.
build build_all_plots='0':
    python scripts/build_plots.py {{build_all_plots}}

# Run checks.
check: check-todos check-unwanted-python-commands

# Check for todos in python files.
check-todos:
    scripts/check-todos.sh

# Check for print statements and similar in python files in python directory.
check-unwanted-python-commands:
    scripts/check-unwanted-python-commands.sh

# Clean data
[confirm]
clean-data subdir='__everything__':
    rm -rf {{ if subdir == '__everything__' {"data"} else{"data/" + subdir} }}

# Clean data with filename pattern
[confirm]
clean-data-pattern pattern:
    find data/ -type f -name '{{ pattern }}' -exec rm {} +

# Clean plots with filename pattern
[confirm]
clean-plots-pattern pattern:
    find tex/plots/ -type f -name '{{ pattern }}' -exec rm {} +

# Clean data and plots with filename pattern
clean-pattern pattern:
    @just clean-plots-pattern {{pattern}}
    @just clean-data-pattern {{pattern}}

# Clean the coverage files
clean:
    rm -rf .coverage
    rm -rf htmlcov

# Clean all data and build and test and check files
clean-all: clean clean-data

# Generate test data.
generate-test-data:
    python scripts/generate_test_data.py

# Profile how many cores are optimal for our computations
profile-parallel:
    python scripts/profile_parallel.py

_requirements-pip:
    python -m pip install --upgrade pip

# Install python requirements from the target directory
requirements target='python': _requirements-pip
    python -m pip install -r {{ target }}/requirements.txt

# Install python requirements from all directories
requirements-all: _requirements-pip
    for f in `ls ./**/requirements.txt | grep -v 'docker/requirements.txt'`; do \
        python -m pip install -r $f; \
    done

# Run python tests
test-code:
    python -m coverage run --branch -m pytest test

# Test if test coverage is ok
test-coverage: test-code
    scripts/test-coverage.sh

# Run the linter
test-lint:
    pylint --disable=C,R,W0511 python scripts

# Run the type checker
test-typecheck:
    mypy python scripts

# Run all python tests
test: test-coverage test-lint test-typecheck

# Change permission of file in git
git-scripts-executable:
    git update-index  --add --chmod=+x scripts/*.sh
    git commit -m "Make scripts executable in git."


