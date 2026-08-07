"""Pytest configuration for local imports.

Ensure the repository root is on sys.path so tests can import top-level modules
like buffer.py, loop.py, and evaluator.py consistently under pytest.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
