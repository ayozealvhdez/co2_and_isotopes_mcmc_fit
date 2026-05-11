"""
Test context helpers.

This module makes the project root importable when individual test files are
executed directly from the command line or through unittest discovery.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
