#!/usr/bin/env python3

"""
Script to check SSH data directly from retrieve_sat.py
"""

from retrieve_sat import retrieve_satellite_data
import numpy as np
from astropy.time import Time

def main():
    # Example query
    queries = [(45.0, -30.0, 2459020.5)]
    
    # Only SSH product
    products = {
        "ssh": ["adt", "sla", "ugos", "vgos"]
    }
    
    # Parameters
    spatial_padding = 16
    temporal_padding = 4
    
    # Retrieve data
    print("Retrieving SSH data...")
    results = retrieve_satellite_data(queries, products, spatial_padding, temporal_padding)
    
    # Examine SSH data
    ssh_data = results[0]['ssh']
    print("\nSSH data structure:")
    for key in ssh_data.keys():
        print(f"  {key}")
    
    print("\nCoordinates:")
    for coord, values in ssh_data['coordinates'].items():
        print(f"  {coord}: shape={np.shape(values)}")
    
    print("\nData dimensions:")
    for var, data in ssh_data['data'].items():
        print(f"  {var}: shape={np.shape(data)}, dtype={getattr(data, 'dtype', None)}")
        
        # Show time values if present
        if var == 'time':
            print(f"    values: {data}")

if __name__ == "__main__":
    main() 