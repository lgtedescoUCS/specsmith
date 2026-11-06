"""Run the evaluation. See `uv run python eval/run_eval.py --help`.

Real runs call the LLM and spend tokens. `--rescore REPORT` scores a
stored report again without calling it.
"""

import sys

from specsmith.evaluation.cli import main

if __name__ == "__main__":
    sys.exit(main())
