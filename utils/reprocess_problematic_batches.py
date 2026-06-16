#!/usr/bin/env python3
import os
import h5py
import argparse
import subprocess
from tqdm import tqdm


def read_problematic_batches(log_file):
    """Read the problematic batches log file and return a list of problematic files."""
    problematic_files = []
    current_file = None
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('File:'):
                current_file = line.split('File:')[1].strip()
                problematic_files.append(current_file)
    
    return problematic_files


def extract_stations_from_batch(batch_file):
    """Extract station information from a batch file."""
    stations = []
    try:
        with h5py.File(batch_file, 'r') as f:
            if 'stations' not in f:
                return stations
            
            st_group = f['stations']
            n_stations = len(st_group['latitude'])
            
            for i in range(n_stations):
                station = {
                    'latitude': st_group['latitude'][i],
                    'longitude': st_group['longitude'][i],
                    'julian_date': st_group['julian_date'][i],
                    'source_file': st_group['source_file'][i],
                    'profile_index': st_group['profile_index'][i]
                }
                stations.append(station)
    except Exception as e:
        print(f"Error reading {batch_file}: {e}")
    
    return stations


def reprocess_batches(log_file, output_dir, spatial_padding, temporal_padding):
    """Reprocess the problematic batches identified in the log file."""
    # Read problematic files
    problematic_files = read_problematic_batches(log_file)
    if not problematic_files:
        print("No problematic files found in the log.")
        return
    
    # Extract all stations from problematic batches
    all_stations = []
    for batch_file in tqdm(problematic_files, desc="Extracting stations"):
        stations = extract_stations_from_batch(batch_file)
        all_stations.extend(stations)
    
    if not all_stations:
        print("No stations found in problematic batches.")
        return
    
    # Create a temporary file with station information
    temp_file = os.path.join(output_dir, "stations_to_reprocess.txt")
    with open(temp_file, 'w') as f:
        for station in all_stations:
            f.write(f"{station['latitude']} {station['longitude']} {station['julian_date']}\n")
    
    # Call the original satellite data generation script
    cmd = [
        "python", "generate_satellite_data.py",
        "--stations_file", temp_file,
        "--output_dir", output_dir,
        "--spatial_padding", str(spatial_padding),
        "--temporal_padding", str(temporal_padding)
    ]
    
    print("Reprocessing stations...")
    subprocess.run(cmd)
    
    # Clean up temporary file
    os.remove(temp_file)
    print("Reprocessing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reprocess problematic satellite data batches.")
    parser.add_argument('--log_file', type=str, required=True,
                        help="Path to the problematic_batches.log file")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save reprocessed data")
    parser.add_argument('--spatial_padding', type=int, default=16,
                        help="Spatial padding for satellite data retrieval")
    parser.add_argument('--temporal_padding', type=int, default=1,
                        help="Temporal padding for satellite data retrieval")
    
    args = parser.parse_args()
    reprocess_batches(args.log_file, args.output_dir, args.spatial_padding, args.temporal_padding) 