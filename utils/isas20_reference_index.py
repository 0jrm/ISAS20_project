#!/usr/bin/env python3
import os
import glob
import pickle
import argparse
from netCDF4 import Dataset
from tqdm import tqdm

def build_index(data_dir):
    """
    Build an index mapping station identifier (lat, lon, julian_date) to a record
    containing the netCDF file name and profile index.
    """
    index = {}
    # Julian date settings
    JD_OFFSET = 2433282.5  # ARGO JULD is defined as days since 1950-01-01
    
    # Collect all .nc files from the provided directory.
    nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    if not nc_files:
        print(f"No .nc files found in {data_dir}")
        return index

    # Loop through each file with progress indication.
    for nc_file in tqdm(nc_files, desc="Processing files"):
        try:
            ds = Dataset(nc_file, mode='r')
        except Exception as e:
            print(f"Error opening {nc_file}: {e}")
            continue

        try:
            # Using the correct variable names from the netCDF file
            lats = ds.variables["LATITUDE"][:]
            lons = ds.variables["LONGITUDE"][:]
            julian_dates = ds.variables["JULD"][:]
            julian_dates = julian_dates + JD_OFFSET  # Convert to astronomical Julian days
        except Exception as e:
            print(f"Error reading variables from {nc_file}: {e}")
            ds.close()
            continue

        # Assume that each file has N_PROF stations and iterate over them.
        for i in range(len(lats)):
            key = (float(lats[i]), float(lons[i]), int(julian_dates[i]))
            # Store additional metadata to retrieve station data later.
            record = {"file": nc_file, "profile_index": i}
            index[key] = record

        ds.close()
    return index

def main():
    parser = argparse.ArgumentParser(
        description="Build a reference index for ISAS20_ARGO netCDF station data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/Processed/TS_merged",
        help="Directory containing the netCDF files."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="isas20_reference_index.pkl",
        help="Path to save the reference index."
    )
    args = parser.parse_args()

    # Build the index.
    index = build_index(args.data_dir)
    # Save the index efficiently using pickle.
    with open(args.output, "wb") as f:
        pickle.dump(index, f)
    print(f"Index built with {len(index)} entries and saved to {args.output}")

if __name__ == "__main__":
    main()