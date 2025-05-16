# Usage examples

'''
# Combine files but keep the batch files
python combine_satellite_data.py \
    --output_dir ./NeSPReSO_v2_GoM_sat \
    --combined_file satellite_NeSPReSO_v2_GoM.h5

# Combine files and delete the batch files afterward
python combine_satellite_data.py \
    --output_dir ./NeSPReSO_v1_GoM_sat \
    --combined_file satellite_NeSPReSO_v2_GoM.h5 \
    --delete_batches
'''

import os
import glob
import h5py
import numpy as np
import argparse
from tqdm import tqdm


def check_batch_dimensions(batch_files):
    """
    Check dimensions of datasets in each batch file and identify problematic ones.
    Returns a list of problematic batch files and their issues.
    """
    problematic_batches = []
    
    # First pass: collect all valid dimensions for each variable
    print("\nAnalyzing dimensions across all batches...")
    var_dimensions = {}  # (group, dataset) -> set of shapes
    
    for batch_file in tqdm(batch_files, desc="First pass: Analyzing dimensions"):
        try:
            with h5py.File(batch_file, "r") as f:
                for group_name in f:
                    if group_name in ['metadata', 'stations', 'problematic_stations']:
                        continue
                    
                    prod_group = f[group_name]
                    for ds_name in prod_group:
                        data = prod_group[ds_name][:]
                        key = (group_name, ds_name)
                        if key not in var_dimensions:
                            var_dimensions[key] = set()
                        var_dimensions[key].add(data.shape)
        except Exception as e:
            problematic_batches.append({
                'file': batch_file,
                'error': str(e)
            })
    
    print(f"Found {len(var_dimensions)} unique variables across all batches")
    
    # Second pass: identify batches with inconsistent dimensions
    print("\nChecking for consistency across batches...")
    for batch_file in tqdm(batch_files, desc="Second pass: Checking consistency"):
        try:
            with h5py.File(batch_file, "r") as f:
                for group_name in f:
                    if group_name in ['metadata', 'stations', 'problematic_stations']:
                        continue
                    
                    prod_group = f[group_name]
                    for ds_name in prod_group:
                        data = prod_group[ds_name][:]
                        key = (group_name, ds_name)
                        if data.shape not in var_dimensions[key]:
                            problematic_batches.append({
                                'file': batch_file,
                                'group': group_name,
                                'dataset': ds_name,
                                'shape': data.shape,
                                'expected_shapes': list(var_dimensions[key])
                            })
        except Exception as e:
            if batch_file not in [p['file'] for p in problematic_batches]:
                problematic_batches.append({
                    'file': batch_file,
                    'error': str(e)
                })
    
    return problematic_batches


def combine_satellite_files(output_dir, combined_file_name, delete_batches=False):
    # Search for all batch files in the output directory
    print(f"\nSearching for batch files in {output_dir}...")
    batch_pattern = os.path.join(output_dir, "satellite_data_batch_*.h5")
    batch_files = sorted(glob.glob(batch_pattern))
    if not batch_files:
        print(f"No batch files found in {output_dir} matching pattern satellite_data_batch_*.h5")
        return

    print(f"Found {len(batch_files)} batch files to process")

    # First check for problematic batches
    problematic_batches = check_batch_dimensions(batch_files)
    
    if problematic_batches:
        print(f"\nFound {len(problematic_batches)} problematic batches")
        # Create a log file for problematic batches
        log_file = os.path.join(output_dir, "problematic_batches.log")
        print(f"Writing details to {log_file}...")
        with open(log_file, "w") as f:
            f.write("Problematic batches found:\n\n")
            for problem in problematic_batches:
                if 'error' in problem:
                    f.write(f"File: {problem['file']}\n")
                    f.write(f"Error: {problem['error']}\n\n")
                else:
                    f.write(f"File: {problem['file']}\n")
                    f.write(f"Group: {problem['group']}\n")
                    f.write(f"Dataset: {problem['dataset']}\n")
                    f.write(f"Shape: {problem['shape']}\n")
                    f.write(f"Expected shapes: {problem['expected_shapes']}\n\n")
        
        # Remove problematic batch files
        problematic_files = set(p['file'] for p in problematic_batches)
        print("\nRemoving problematic batch files...")
        for batch_file in problematic_files:
            try:
                os.remove(batch_file)
                print(f"Removed problematic batch: {batch_file}")
            except Exception as e:
                print(f"Failed to remove {batch_file}: {e}")
        
        # Update batch_files list
        batch_files = sorted(glob.glob(batch_pattern))
        if not batch_files:
            print("No valid batch files remaining after removing problematic ones.")
            return
        print(f"\n{len(batch_files)} valid batch files remaining")

    # Initialize containers for stations and product groups
    print("\nInitializing data containers...")
    stations_data = {}
    products_data = {}
    problematic_data = {}

    # Process each batch file
    print("\nProcessing batch files and collecting data...")
    for batch_file in tqdm(batch_files, desc="Reading batch files"):
        with h5py.File(batch_file, "r") as f:
            # Process stations group
            st_group = f['stations']
            for key in st_group:
                if key not in stations_data:
                    stations_data[key] = []
                stations_data[key].append(st_group[key][:])

            # Process all groups that are not 'metadata', 'stations', or 'problematic_stations'
            for group_name in f:
                if group_name in ['metadata', 'stations', 'problematic_stations']:
                    continue
                if group_name not in products_data:
                    products_data[group_name] = {}
                prod_group = f[group_name]
                for ds_name in prod_group:
                    if ds_name not in products_data[group_name]:
                        products_data[group_name][ds_name] = []
                    products_data[group_name][ds_name].append(prod_group[ds_name][:])

            # Process problematic_stations group if exists
            if 'problematic_stations' in f:
                prob_group = f['problematic_stations']
                for key in prob_group:
                    if key not in problematic_data:
                        problematic_data[key] = []
                    problematic_data[key].append(prob_group[key][:])

    print(f"\nFound {len(products_data)} product groups and {len(stations_data)} station variables")

    # Concatenate arrays for stations
    print("\nConcatenating station data...")
    combined_stations = {}
    for key, arrays in tqdm(stations_data.items(), desc="Processing station variables"):
        combined_stations[key] = np.concatenate(arrays, axis=0)

    # Concatenate arrays for product groups
    print("\nConcatenating product data...")
    combined_products = {}
    for product, ds_dict in tqdm(products_data.items(), desc="Processing product groups"):
        combined_products[product] = {}
        for ds_name, arrays in tqdm(ds_dict.items(), desc=f"Processing {product} variables", leave=False):
            combined_products[product][ds_name] = np.concatenate(arrays, axis=0)

    # Concatenate problematic stations if any
    if problematic_data:
        print("\nConcatenating problematic station data...")
        combined_problematic = {}
        for key, arrays in tqdm(problematic_data.items(), desc="Processing problematic stations"):
            combined_problematic[key] = np.concatenate(arrays, axis=0)

    # Write the combined data to a new H5 file
    combined_file_path = os.path.join(output_dir, combined_file_name)
    print(f"\nWriting combined data to {combined_file_path}...")
    with h5py.File(combined_file_path, "w") as f:
        # Save stations group
        print("Writing station data...")
        st_group = f.create_group('stations')
        for key, data in tqdm(combined_stations.items(), desc="Saving station variables"):
            st_group.create_dataset(key, data=data)

        # Save product groups
        print("\nWriting product data...")
        for product, ds_dict in tqdm(combined_products.items(), desc="Saving product groups"):
            prod_group = f.create_group(product)
            for ds_name, data in tqdm(ds_dict.items(), desc=f"Saving {product} variables", leave=False):
                prod_group.create_dataset(ds_name, data=data, compression='gzip', compression_opts=9)

        # Save problematic_stations group if available
        if problematic_data:
            print("\nWriting problematic station data...")
            prob_group = f.create_group('problematic_stations')
            for key, data in tqdm(combined_problematic.items(), desc="Saving problematic stations"):
                prob_group.create_dataset(key, data=data)

    print(f"\nCombined satellite data saved to {combined_file_path}")

    # Delete batch files if requested
    if delete_batches:
        print("\nDeleting batch files...")
        for batch_file in tqdm(batch_files, desc="Removing batch files"):
            try:
                os.remove(batch_file)
            except Exception as e:
                print(f"Failed to remove {batch_file}: {e}")
        print("Batch files removed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine satellite H5 batch files into a single file and remove intermediary files.")
    parser.add_argument('--output_dir', type=str, default='./output', help="Directory containing satellite batch files")
    parser.add_argument('--combined_file', type=str, default='combined_satellite_data.h5', help="Name of the combined satellite H5 file")
    parser.add_argument('--delete_batches', action='store_true', help="Delete batch files after successful combination")
    args = parser.parse_args()
    combine_satellite_files(args.output_dir, args.combined_file, args.delete_batches) 