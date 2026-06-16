#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def inspect_h5_file(file_path):
    """Inspect the contents of an HDF5 file and print shapes and time information."""
    print(f"\nInspecting file: {file_path}")
    with h5py.File(file_path, 'r') as h5f:
        # Print identification array shape
        print("\nIdentification array shape:", h5f['identification'].shape)
        
        # Inspect satellite data
        sat_grp = h5f['satellite_data']
        for query_idx, query_grp in sat_grp.items():
            print(f"\nQuery {query_idx}:")
            print(f"Latitude: {query_grp.attrs['latitude']}")
            print(f"Longitude: {query_grp.attrs['longitude']}")
            print(f"Julian Date: {query_grp.attrs['julian_date']}")
            
            for prod_name, prod_grp in query_grp.items():
                print(f"\nProduct: {prod_name}")
                print(f"Source file: {prod_grp.attrs['file']}")
                
                # Print coordinate shapes
                if 'latitude' in prod_grp:
                    print(f"Latitude shape: {prod_grp['latitude'].shape}")
                if 'longitude' in prod_grp:
                    print(f"Longitude shape: {prod_grp['longitude'].shape}")
                
                # Print data shapes and time information
                data_grp = prod_grp['data']
                for var_name, var_data in data_grp.items():
                    print(f"{var_name} shape: {var_data.shape}")
                    if var_name == 'time':
                        # Convert timestamps back to datetime for readability
                        times = [datetime.fromtimestamp(ts) for ts in var_data[:]]
                        print(f"Time range: {times[0]} to {times[-1]}")
                        print(f"Number of time steps: {len(times)}")

def plot_ssh_data(file_path, query_idx=0):
    """Plot SSH data for a specific query to visualize the fix."""
    with h5py.File(file_path, 'r') as h5f:
        query_grp = h5f[f'satellite_data/query_{query_idx}']
        ssh_grp = query_grp['ssh']
        data_grp = ssh_grp['data']
        
        # Get time array
        times = [datetime.fromtimestamp(ts) for ts in data_grp['time'][:]]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'SSH Data for Query {query_idx}\n'
                    f'Lat: {query_grp.attrs["latitude"]:.2f}, '
                    f'Lon: {query_grp.attrs["longitude"]:.2f}')
        
        # Plot each variable
        for idx, (var_name, ax) in enumerate(zip(['ADT', 'SLA', 'UGOS', 'VGOS'], axes.flat)):
            data = data_grp[var_name][:]
            # Plot mean over spatial dimensions
            mean_data = np.mean(data, axis=(1, 2))
            ax.plot(times, mean_data)
            ax.set_title(var_name)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('ssh_data_test.png')
        plt.close()

def main():
    # Test with a small batch file
    batch_dir = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/v2/sat_inputs_batches"
    batch_files = sorted([f for f in os.listdir(batch_dir) if f.startswith('batch_')])
    
    if not batch_files:
        print("No batch files found!")
        return
    
    # Test with the first batch file
    test_file = os.path.join(batch_dir, batch_files[0])
    inspect_h5_file(test_file)
    plot_ssh_data(test_file)

if __name__ == "__main__":
    main() 