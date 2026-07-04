import os
import sys
from datetime import datetime
import copernicusmarine

# Directory to save SSH files
output_dir = "/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSH"

# Dataset info for SSH
DATASET_ID = "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D"
DATASET_VERSION = "202411"

# Years to download (override via CLI: python download_SSH.py 2021 2022)
YEARS = [int(y) for y in sys.argv[1:]] or [1999]

# Download a single year
def download_year(year, output_dir):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    date_str = f"{year}"
    try:
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            variables=[
                "adt", "sla", "tpa_correction", "ugos", "ugosa", "vgos", "vgosa", "flag_ice", "err_vgosa", "err_ugosa", "err_sla"
            ],
            minimum_longitude=-179.875,
            maximum_longitude=179.875,
            minimum_latitude=-89.875,
            maximum_latitude=89.875,
            start_datetime=start_date.strftime("%Y-%m-%dT00:00:00"),
            end_datetime=end_date.strftime("%Y-%m-%dT23:59:59"),
            output_directory=output_dir,
            output_filename=f"SSH_{date_str}.nc",
            disable_progress_bar=False
        )
        print(f"Downloaded SSH_{date_str}.nc")
        return True
    except Exception as e:
        print(f"Failed to download SSH_{date_str}.nc: {e}")
        return False

def main():
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    success = 0
    for year in YEARS:
        total += 1
        if download_year(year, output_dir):
            success += 1
    print(f"\nDownload summary: {success}/{total} files downloaded successfully.")

if __name__ == "__main__":
    main()
