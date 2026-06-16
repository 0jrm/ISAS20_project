import os
import copernicusmarine
from datetime import datetime
import re
from pathlib import Path

def extract_date_from_filename(filename):
    """Extract date from filename like SSS_YYYYMMDD.nc"""
    match = re.search(r'SSS_(\d{8})\.nc', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    return None

def download_file(date, output_dir):
    """Download a single file for a specific date"""
    try:
        # Format date for the API
        date_str = date.strftime('%Y-%m-%d')
        
        # Download the file
        copernicusmarine.subset(
            dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
            dataset_version="202311",
            # variables=["sos", "sos_error"],  # Only downloading SSS variables
            minimum_longitude=-179.9375,
            maximum_longitude=179.9375,
            minimum_latitude=-89.9375,
            maximum_latitude=89.9375,
            start_datetime=f"{date_str}T00:00:00",
            end_datetime=f"{date_str}T00:00:00",
            minimum_depth=0,
            maximum_depth=0,
            # coordinates_selection_method="strict-inside",
            # netcdf_compression_level=1,
            disable_progress_bar=False,
            output_directory=output_dir,
            output_filename=f"SSS_{date.strftime('%Y%m%d')}.nc",
            force_download=True  # Automatically accept download prompts
        )
        print(f"Successfully downloaded file for {date_str}")
        return True
    except Exception as e:
        print(f"Error downloading file for {date_str}: {str(e)}")
        return False

def main():
    # Directory containing the corrupt files list and where to save new downloads
    data_dir = "/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS"
    corrupt_files_path = os.path.join(data_dir, "corrupt_files.txt")
    
    # Read the list of corrupt files
    with open(corrupt_files_path, 'r') as f:
        corrupt_files = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(corrupt_files)} corrupt files to process")
    
    # Process each file
    successful_downloads = 0
    failed_downloads = 0
    
    for file_path in corrupt_files:
        date = extract_date_from_filename(file_path)
        if date:
            print(f"\nProcessing file: {os.path.basename(file_path)}")
            if download_file(date, data_dir):
                successful_downloads += 1
            else:
                failed_downloads += 1
        else:
            print(f"Could not extract date from filename: {file_path}")
            failed_downloads += 1
    
    print("\nDownload Summary:")
    print(f"Total files processed: {len(corrupt_files)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")

if __name__ == "__main__":
    main()