"""Make the gate module importable when tests run from the repo root
(`python -m pytest code/gate/tests`): put code/gate on sys.path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
