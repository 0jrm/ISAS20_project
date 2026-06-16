#!/usr/bin/env python3
import os
import h5py
import numpy as np
import argparse
import netCDF4
from tqdm import tqdm

# Example usage:
# python gather_model_output.py --combined_sat_file /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/utils/NeSPReSO_v1_global_sat/satellite_NeSPReSO_v1_global.h5 --output_file /unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/utils/NeSPReSO_v1_global_sat/NeSPReSO_v1_global_ISAS_data.h5

def gather_model_output(combined_sat_file, output_file):
    """
    Gather model output data from ISAS20 NetCDF files based on satellite availability.
    Reads the combined satellite H5 file to get station metadata (including source netcdf files and profile indices), 
    then extracts variables from each netcdf file for the specified profile and saves the aggregated data in an output H5 file.
    """
    # Open the combined satellite file and extract station metadata
    with h5py.File(combined_sat_file, 'r') as sat_f:
        stations_grp = sat_f['stations']
        lat_array = stations_grp['latitude'][:]
        lon_array = stations_grp['longitude'][:]
        jd_array = stations_grp['julian_date'][:]
        src_array = stations_grp['source_file'][:]
        profile_indices = stations_grp['profile_index'][:]

    # Convert source file entries to strings if necessary
    src_files = []
    for s in src_array:
        if isinstance(s, bytes):
            src_files.append(s.decode('utf-8'))
        else:
            src_files.append(s)

    N = len(lat_array)
    # Create a list to store model outputs for each station
    # Each element is a dict mapping variable name -> extracted data for that station
    model_outputs = [{} for _ in range(N)]

    # Group station indices by source netcdf file
    file_groups = {}
    for i, src in enumerate(src_files):
        file_groups.setdefault(src, []).append((i, int(profile_indices[i])))

    # Process each unique netcdf file
    for src, indices in tqdm(file_groups.items(), desc='Processing NetCDF files'):
        if not os.path.exists(src):
            print(f"NetCDF file {src} not found, skipping stations from this file.")
            continue
        try:
            ds = netCDF4.Dataset(src, 'r')
        except Exception as e:
            print(f"Error opening NetCDF file {src}: {e}")
            continue

        # Determine number of profiles using a common variable if available
        if 'LATITUDE' in ds.variables:
            num_profiles = ds.variables['LATITUDE'].shape[0]
        else:
            num_profiles = None

        # Identify variables to extract: those with at least one dimension and whose first dimension equals num_profiles
        valid_vars = []
        for var_name, var_obj in ds.variables.items():
            if num_profiles is not None and var_obj.ndim >= 1 and var_obj.shape[0] == num_profiles:
                valid_vars.append(var_name)

        # For each station corresponding to this netcdf file, extract data for each valid variable
        for overall_idx, prof_idx in indices:
            if num_profiles is not None and (prof_idx < 0 or prof_idx >= num_profiles):
                print(f"Invalid profile index {prof_idx} for file {src}, skipping station {overall_idx}.")
                continue
            for var_name in valid_vars:
                try:
                    data = ds.variables[var_name][prof_idx]
                    # Convert masked arrays to regular arrays with fill value np.nan
                    if hasattr(data, 'mask'):
                        data = np.array(data.filled(np.nan))
                    else:
                        data = np.array(data)
                    model_outputs[overall_idx][var_name] = data
                except Exception as e:
                    print(f"Error reading variable {var_name} for profile {prof_idx} from file {src}: {e}")
        ds.close()

    # Determine the union of all variable names and record a sample shape and dtype
    var_info = {}  # var_name -> (shape, dtype)
    for station_data in model_outputs:
        for var_name, data in station_data.items():
            if var_name not in var_info:
                var_info[var_name] = (np.shape(data), data.dtype)

    # For each variable in the union, create a list of station data, filling missing entries with default values
    combined_vars = {}
    for var_name, (shape, dtype) in var_info.items():
        data_list = [None] * N
        for i in range(N):
            if var_name in model_outputs[i]:
                data_list[i] = model_outputs[i][var_name]
            else:
                # Fill with default: np.nan for floats, -9999 for integers, empty string for others
                if np.issubdtype(dtype, np.floating):
                    data_list[i] = np.full(shape, np.nan, dtype=dtype)
                elif np.issubdtype(dtype, np.integer):
                    data_list[i] = np.full(shape, -9999, dtype=dtype)
                else:
                    # For string data, use ASCII encoding with fixed length
                    max_len = max(len(str(x)) for x in data_list if x is not None) if any(x is not None for x in data_list) else 1
                    data_list[i] = np.full(shape, '', dtype=f'S{max_len}')
        try:
            combined_vars[var_name] = np.stack(data_list, axis=0)
        except Exception as e:
            print(f"Error stacking data for variable {var_name}: {e}")

    # Write the aggregated model output data to the output H5 file
    with h5py.File(output_file, 'w') as out_f:
        # Save station metadata in group 'stations'
        st_grp = out_f.create_group('stations')
        st_grp.create_dataset('latitude', data=lat_array)
        st_grp.create_dataset('longitude', data=lon_array)
        st_grp.create_dataset('julian_date', data=jd_array)
        # Convert source files to ASCII strings with fixed length
        max_len = max(len(s) for s in src_files)
        src_array_ascii = np.array([s.encode('ascii', 'ignore') for s in src_files], dtype=f'S{max_len}')
        st_grp.create_dataset('source_file', data=src_array_ascii)
        st_grp.create_dataset('profile_index', data=profile_indices)

        # Save model output variables in group 'model'
        model_grp = out_f.create_group('model')
        for var_name, data in combined_vars.items():
            # Handle string data specially
            if data.dtype.kind in ['U', 'S']:
                # Convert to ASCII if Unicode
                if data.dtype.kind == 'U':
                    data = np.char.encode(data, 'ascii', 'ignore')
                model_grp.create_dataset(var_name, data=data)
            else:
                model_grp.create_dataset(var_name, data=data, compression='gzip', compression_opts=9)

    print(f"Combined model output data saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather model output data from ISAS20 NetCDF files based on satellite availability.')
    parser.add_argument('--combined_sat_file', type=str, default='./output/combined_satellite_data.h5',
                        help='Path to the combined satellite H5 file')
    parser.add_argument('--output_file', type=str, default='combined_model_output.h5',
                        help='Name of the output combined model H5 file')
    args = parser.parse_args()
    gather_model_output(args.combined_sat_file, args.output_file) 