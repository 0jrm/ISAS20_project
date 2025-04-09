#!/usr/bin/env python
"""
Test script to verify module imports are working correctly
"""

import sys
import os

# Add the parent directory to the Python path so we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Now try to import the module
try:
    from ISAS20_project.utils import retrieve_satellite_data
    print("✓ Successfully imported retrieve_satellite_data")
except ImportError as e:
    print(f"✗ Failed to import: {e}")

# Print the module's docstring
try:
    from ISAS20_project.utils.retrieve_sat import retrieve_satellite_data as rsd
    print("\nModule docstring:")
    print(rsd.__doc__)
except Exception as e:
    print(f"Error getting docstring: {e}")

print("\nImport test complete.") 