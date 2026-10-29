#!/usr/bin/env bash
# Reproduce every table in README.md. Expects data/nq and data/tqqq (see data/README.md).
set -euo pipefail
mkdir -p results
python3 experiments/00_validate.py data/nq            > results/00_validate_nq.txt
python3 experiments/01_nq_robustness.py data/nq       > results/01_nq_robustness.txt
python3 experiments/02_nq_walkforward.py data/nq      > results/02_nq_walkforward.txt
python3 experiments/10_validate_tqqq.py data/tqqq     > results/10_validate_tqqq.txt
python3 experiments/11_tqqq_robustness.py data/tqqq   > results/11_tqqq_robustness.txt
python3 experiments/12_tqqq_walkforward.py data/tqqq  > results/12_tqqq_walkforward.txt
python3 experiments/13_tqqq_raw_cost.py data/tqqq    > results/13_tqqq_raw_cost.txt
python3 experiments/14_paper_spec_comparison.py data/tqqq data/nq > results/14_paper_spec_comparison.txt
echo done
