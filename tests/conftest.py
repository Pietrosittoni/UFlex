import os
import sys

# Make the repository root importable when running `pytest` from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
