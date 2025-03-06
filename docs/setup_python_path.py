"""
Helper script to set up the Python path for quartodoc.
This script adds the parent directory to the Python path so that
quartodoc can find and import the starsim module.
"""
import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Print the Python path for debugging
print("Python path:")
for path in sys.path:
    print(f"  {path}")

# Try to import starsim to verify it works
try:
    import starsim
    print(f"Successfully imported starsim from {starsim.__file__}")
except ImportError as e:
    print(f"Failed to import starsim: {e}")
