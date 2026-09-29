# SPDX-License-Identifier: Apache-2.0
"""Make code/scout and this tests dir importable (`python -m pytest code/scout/tests`)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
