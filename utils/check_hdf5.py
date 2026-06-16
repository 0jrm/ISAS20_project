#!/usr/bin/env python3

"""
Script to check the contents of satellite input HDF5 files
"""

import h5py
import numpy as np
import sys
import os

# File to check
batch_file = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/v2/sat_inputs_test_batches/batch_0004.h5"

def check_hdf5_file(file_path):
    """Check the contents of an HDF5 file for satellite data issues."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print(f"Root keys: {list(f.keys())}")
        
        # List satellite data queries
        sat_data = f['satellite_data']
        queries = list(sat_data.keys())
        print(f"Number of queries: {len(queries)}")
        
        # Examine the first query in detail
        query_key = queries[0]
        query = sat_data[query_key]
        print(f"\nExamining query: {query_key}")
        
        # List products
        products = list(query.keys())
        print(f"Products: {products}")
        
        # Check each product
        for product in products:
            print(f"\n=== Product: {product} ===")
            prod_group = query[product]
            
            # Show basic info
            print(f"File source: {prod_group.attrs.get('file', 'Not specified')}")
            
            # Check coordinates
            if 'latitude' in prod_group:
                print(f"Latitude shape: {prod_group['latitude'].shape}")
            if 'longitude' in prod_group:
                print(f"Longitude shape: {prod_group['longitude'].shape}")
            
            # Check data variables
            data_group = prod_group['data']
            vars = list(data_group.keys())
            print(f"Variables: {vars}")
            
            # Check time information
            if 'time' in vars:
                time_data = data_group['time']
                print(f"Time shape: {time_data.shape}")
                print(f"Time values: {time_data[:]}")
            
            # Sample a few data variables
            for var in vars:
                if var == 'time':
                    continue
                    
                data = data_group[var]
                print(f"\nVariable: {var}")
                print(f"  Shape: {data.shape}")
                print(f"  Data type: {data.dtype}")
                
                # Check values across time dimension for 3D data
                if len(data.shape) == 3:
                    center_x, center_y = data.shape[2] // 2, data.shape[1] // 2
                    print(f"  Values at center point ({center_y}, {center_x}) across time:")
                    for t in range(data.shape[0]):
                        print(f"    t={t}: {data[t, center_y, center_x]}")
                    
                    # Check if data is identical across time
                    identical = True
                    for t in range(1, data.shape[0]):
                        if not np.array_equal(data[0], data[t]):
                            identical = False
                            break
                    print(f"  Data is {'identical' if identical else 'different'} across time steps")
                
                # For 2D data
                elif len(data.shape) == 2:
                    center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
                    print(f"  Value at center point ({center_y}, {center_x}): {data[center_y, center_x]}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_hdf5_file(sys.argv[1])
    else:
        check_hdf5_file(batch_file) 