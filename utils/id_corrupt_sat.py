import os
import xarray as xr
from pathlib import Path
import logging

def check_netcdf_file(file_path):
    """
    Check if a NetCDF file is corrupted by attempting to open it.
    
    Args:
        file_path (str): Path to the NetCDF file
        
    Returns:
        bool: True if file is corrupted, False if file is valid
    """
    try:
        with xr.open_dataset(file_path) as ds:
            # Try to access a basic attribute to ensure file is readable
            _ = ds.attrs
        return False
    except Exception as e:
        logging.warning(f"Error reading file {file_path}: {str(e)}")
        return True

def find_corrupt_files(directory):
    """
    Find all corrupted NetCDF files in a directory.
    
    Args:
        directory (str): Path to directory containing NetCDF files
        
    Returns:
        list: List of paths to corrupted files
    """
    corrupt_files = []
    directory = Path(directory)
    
    # Find all .nc files in the directory
    nc_files = list(directory.glob('**/*.nc'))
    
    print(f"Found {len(nc_files)} NetCDF files to check...")
    
    for file_path in nc_files:
        if check_netcdf_file(str(file_path)):
            corrupt_files.append(str(file_path))
            print(f"Corrupt file found: {file_path}")
    
    return corrupt_files

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Directory containing NetCDF files
    data_dir = "/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS"
    
    print("Starting corruption check...")
    corrupt_files = find_corrupt_files(data_dir)
    
    print("\nSummary:")
    print(f"Total corrupt files found: {len(corrupt_files)}")
    if corrupt_files:
        print("\nList of corrupt files:")
        for file in corrupt_files:
            print(file)
    
    # Save the list of corrupt files to a text file in the same directory
    output_file = os.path.join(data_dir, "corrupt_files.txt")
    with open(output_file, 'w') as f:
        for file in corrupt_files:
            f.write(f"{file}\n")
    print(f"\nList of corrupt files has been saved to {output_file}")

if __name__ == "__main__":
    main()
