#!/usr/bin/env python3
import os
import pickle
import argparse
import h5py
import numpy as np
from datetime import datetime
from astropy.time import Time
from tqdm import tqdm
import warnings
from retrieve_sat import retrieve_satellite_data

def load_and_filter_index(index_path, start_date, end_date, min_lat, max_lat, min_lon, max_lon):
    """
    Load and filter the ISAS20 reference index based on date range and spatial bounds.
    
    Parameters:
        index_path (str): Path to the reference index pickle file
        start_date (datetime): Start date for filtering
        end_date (datetime): End date for filtering
        min_lat (float): Minimum latitude
        max_lat (float): Maximum latitude
        min_lon (float): Minimum longitude
        max_lon (float): Maximum longitude
        
    Returns:
        list: List of tuples (lat, lon, julian_date, file_path, profile_index)
    """
    # Convert dates to Julian dates
    start_jd = Time(start_date).jd
    end_jd = Time(end_date).jd
    
    # Load the index
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    
    # Filter stations
    filtered_stations = []
    for (lat, lon, jd), record in index.items():
        # Check if station is within spatial bounds
        if (min_lat <= lat <= max_lat and 
            min_lon <= lon <= max_lon and 
            start_jd <= jd <= end_jd):
            filtered_stations.append((lat, lon, jd, record['file'], record['profile_index']))
    
    return filtered_stations

def create_h5_dataset(h5file, name, data, compression="gzip", compression_opts=9):
    """Helper function to create HDF5 dataset with compression"""
    return h5file.create_dataset(name, data=data, compression=compression, 
                                compression_opts=compression_opts)

def save_to_h5(output_path, stations, results, products, args):
    """
    Save the satellite data and metadata to HDF5 format.
    
    Parameters:
        output_path (str): Path to save the HDF5 file
        stations (list): List of (lat, lon, julian_date, file_path, profile_index) tuples
        results (dict): Results from retrieve_satellite_data
        products (dict): Product configuration dictionary
        args (Namespace): Command line arguments
    """
    # Create a log file for problematic stations
    log_dir = os.path.join(os.path.dirname(output_path), "logs")
    os.makedirs(log_dir, exist_ok=True)
    batch_name = os.path.splitext(os.path.basename(output_path))[0]
    problem_log = os.path.join(log_dir, f"{batch_name}_problematic_stations.log")
    
    # Track problematic stations
    problematic_stations = []
    valid_stations = []
    valid_results = {}
    
    # First pass: identify problematic stations
    for i, station in enumerate(stations):
        station_idx = i
        station_data = results.get(station_idx, {})
        is_problematic = False
        problem_details = []
        
        # Check each product and variable
        for product_name in products.keys():
            if product_name not in station_data:
                continue
                
            product_data = station_data[product_name]
            for var_name in products[product_name]:
                if var_name not in product_data['data']:
                    continue
                    
                data = product_data['data'][var_name]
                if not isinstance(data, np.ndarray):
                    is_problematic = True
                    problem_details.append(f"Non-array data for {product_name}/{var_name}")
                    continue
                    
                # Check for expected shape (33, 33) or (t, 33, 33)
                if len(data.shape) not in [2, 3]:
                    is_problematic = True
                    problem_details.append(f"Unexpected shape {data.shape} for {product_name}/{var_name}")
                    continue
                    
                if data.shape[-2:] != (33, 33):
                    is_problematic = True
                    problem_details.append(f"Unexpected spatial dimensions {data.shape[-2:]} for {product_name}/{var_name}")
        
        if is_problematic:
            problematic_stations.append({
                'station': station,
                'index': station_idx,
                'details': problem_details
            })
        else:
            valid_stations.append(station)
            valid_results[station_idx] = station_data
    
    # Log problematic stations
    if problematic_stations:
        with open(problem_log, 'w') as f:
            f.write(f"Problematic stations in batch {batch_name}:\n")
            f.write(f"Total stations: {len(stations)}\n")
            f.write(f"Problematic stations: {len(problematic_stations)}\n")
            f.write(f"Valid stations: {len(valid_stations)}\n\n")
            
            for problem in problematic_stations:
                station = problem['station']
                f.write(f"\nStation {problem['index']}:\n")
                f.write(f"  Latitude: {station[0]}\n")
                f.write(f"  Longitude: {station[1]}\n")
                f.write(f"  Julian Date: {station[2]}\n")
                f.write(f"  Source File: {station[3]}\n")
                f.write(f"  Profile Index: {station[4]}\n")
                f.write("  Problems:\n")
                for detail in problem['details']:
                    f.write(f"    - {detail}\n")
    
    # If no valid stations, skip saving
    if not valid_stations:
        print(f"Warning: No valid stations in batch {batch_name}. Skipping save.")
        return
    
    # Save valid stations to HDF5
    with h5py.File(output_path, 'w') as f:
        # Store metadata
        meta = f.create_group('metadata')
        meta.attrs['start_date'] = str(args.start_date)
        meta.attrs['end_date'] = str(args.end_date)
        meta.attrs['min_lat'] = args.min_lat
        meta.attrs['max_lat'] = args.max_lat
        meta.attrs['min_lon'] = args.min_lon
        meta.attrs['max_lon'] = args.max_lon
        meta.attrs['spatial_padding'] = args.spatial_padding
        meta.attrs['temporal_padding'] = args.temporal_padding
        meta.attrs['products'] = str(products)
        meta.attrs['index_path'] = args.index_path
        meta.attrs['original_batch_size'] = len(stations)
        meta.attrs['valid_stations'] = len(valid_stations)
        meta.attrs['problematic_stations'] = len(problematic_stations)
        
        # Create groups for each product
        for product_name in products.keys():
            product_group = f.create_group(product_name)
            
            # Get the shape of data for this product from the first non-empty result
            sample_data = None
            for res in valid_results.values():
                if product_name in res and res[product_name]['data']:
                    sample_data = res[product_name]
                    break
            
            if sample_data is None:
                continue
                
            # Store coordinates
            coords = sample_data['coordinates']
            product_group.create_dataset('latitude_grid', data=coords['latitude'])
            product_group.create_dataset('longitude_grid', data=coords['longitude'])
            
            # Create datasets for each variable
            for var_name in products[product_name]:
                if var_name in sample_data['data']:
                    var_shape = sample_data['data'][var_name].shape
                    
                    # Create dataset with full shape
                    full_shape = (len(valid_stations),) + var_shape
                    var_ds = product_group.create_dataset(
                        var_name, 
                        shape=full_shape,
                        dtype=np.float32,
                        compression="gzip",
                        compression_opts=9,
                        fillvalue=np.nan
                    )
                    
                    # Fill data
                    for i, (station_idx, station_data) in enumerate(valid_results.items()):
                        if (product_name in station_data and 
                            var_name in station_data[product_name]['data']):
                            var_ds[i] = station_data[product_name]['data'][var_name]
        
        # Store station information
        station_group = f.create_group('stations')
        station_group.create_dataset('latitude', data=[s[0] for s in valid_stations])
        station_group.create_dataset('longitude', data=[s[1] for s in valid_stations])
        station_group.create_dataset('julian_date', data=[s[2] for s in valid_stations])
        station_group.create_dataset('source_file', data=np.array([s[3] for s in valid_stations], dtype='S'))
        station_group.create_dataset('profile_index', data=[s[4] for s in valid_stations])
        
        # Store information about problematic stations
        if problematic_stations:
            problem_group = f.create_group('problematic_stations')
            problem_group.create_dataset('indices', data=[p['index'] for p in problematic_stations])
            problem_group.create_dataset('latitudes', data=[p['station'][0] for p in problematic_stations])
            problem_group.create_dataset('longitudes', data=[p['station'][1] for p in problematic_stations])
            problem_group.create_dataset('julian_dates', data=[p['station'][2] for p in problematic_stations])
            problem_group.create_dataset('source_files', data=np.array([p['station'][3] for p in problematic_stations], dtype='S'))
            problem_group.create_dataset('profile_indices', data=[p['station'][4] for p in problematic_stations])
            
            # Store problem details as attributes
            for i, problem in enumerate(problematic_stations):
                problem_group.attrs[f'station_{i}_details'] = str(problem['details'])

def main():
    parser = argparse.ArgumentParser(
        description="Generate HDF5 files containing satellite data for ISAS20 stations."
    )
    
    # Required arguments
    parser.add_argument('--start_date', type=str, required=True,
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument('--end_date', type=str, required=True,
                        help="End date in YYYY-MM-DD format")
    parser.add_argument('--min_lat', type=float, required=True,
                        help="Minimum latitude")
    parser.add_argument('--max_lat', type=float, required=True,
                        help="Maximum latitude")
    parser.add_argument('--min_lon', type=float, required=True,
                        help="Minimum longitude")
    parser.add_argument('--max_lon', type=float, required=True,
                        help="Maximum longitude")
    parser.add_argument('--spatial_padding', type=int, required=True,
                        help="Spatial padding for satellite data retrieval")
    parser.add_argument('--temporal_padding', type=int, required=True,
                        help="Temporal padding for satellite data retrieval")
    
    # Optional arguments with defaults
    parser.add_argument('--index_path', type=str,
                        default="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/isas20_reference_index.pkl",
                        help="Path to the ISAS20 reference index")
    parser.add_argument('--output_dir', type=str, default="./output",
                        help="Directory to save output HDF5 files")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="Number of stations to process in each batch")
    parser.add_argument('-m', '--missing_batches_only', action='store_true',
                        help="Skip batches whose output file already exists. This is a resume/resume-like feature.")
    
    args = parser.parse_args()
    
    # Convert date strings to datetime objects
    args.start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    args.end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define satellite products and variables
    products = {
        "bathymetry": ["elevation"],
        "ssh": ["adt", "sla", "ugos", "vgos"],
        "ostia": ["analysed_sst"],
        "sss": ["sos"],
        "wind": ["windspeed", "u_wind", "v_wind"]
    }
    
    # Load and filter stations
    print("Loading and filtering ISAS20 reference index...")
    stations = load_and_filter_index(
        args.index_path,
        args.start_date,
        args.end_date,
        args.min_lat,
        args.max_lat,
        args.min_lon,
        args.max_lon
    )
    
    print(f"Found {len(stations)} stations within specified bounds")
    
    # Process stations in batches
    for batch_start in range(0, len(stations), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(stations))
        batch_stations = stations[batch_start:batch_end]
        
        # Generate output filename for this batch
        output_file = os.path.join(
            args.output_dir,
            f"satellite_data_batch_{batch_start:06d}-{batch_end:06d}.h5"
        )
        
        # NEW: Skip if missing_batches_only is set and output file exists
        if args.missing_batches_only and os.path.exists(output_file):
            print(f"Skipping batch {batch_start}-{batch_end} (output file already exists)")
            continue
        
        print(f"\nProcessing batch {batch_start}-{batch_end} of {len(stations)} stations")
        
        # Extract only lat, lon, julian_date for retrieve_satellite_data
        query_stations = [(s[0], s[1], s[2]) for s in batch_stations]
        
        # Retrieve satellite data for this batch
        results = retrieve_satellite_data(
            query_stations,
            products,
            args.spatial_padding,
            args.temporal_padding
        )
        
        # Save results to HDF5
        print(f"Saving results to {output_file}")
        save_to_h5(output_file, batch_stations, results, products, args)
        
        print(f"Completed batch {batch_start}-{batch_end}")

if __name__ == "__main__":
    main() 