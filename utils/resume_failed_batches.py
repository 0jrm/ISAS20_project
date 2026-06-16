#!/usr/bin/env python3
import os
import re
import argparse
import pickle
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
from astropy.time import Time
import logging
import subprocess
from generate_satellite_data import load_and_filter_index, save_to_h5

def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "resume_processing.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_parameters_from_log(log_file):
    """
    Parse parameters from the old log file.
    
    Returns:
        dict: Dictionary containing all parameters
    """
    logger = logging.getLogger(__name__)
    parameters = {}
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Parse date parameters
                if 'start_date:' in line:
                    parameters['start_date'] = datetime.strptime(line.split('start_date: ')[1].strip(), '%Y-%m-%d %H:%M:%S')
                elif 'end_date:' in line:
                    parameters['end_date'] = datetime.strptime(line.split('end_date: ')[1].strip(), '%Y-%m-%d %H:%M:%S')
                
                # Parse spatial parameters
                elif 'min_lat:' in line:
                    parameters['min_lat'] = float(line.split('min_lat: ')[1].strip())
                elif 'max_lat:' in line:
                    parameters['max_lat'] = float(line.split('max_lat: ')[1].strip())
                elif 'min_lon:' in line:
                    parameters['min_lon'] = float(line.split('min_lon: ')[1].strip())
                elif 'max_lon:' in line:
                    parameters['max_lon'] = float(line.split('max_lon: ')[1].strip())
                
                # Parse padding parameters
                elif 'spatial_padding:' in line:
                    parameters['spatial_padding'] = int(line.split('spatial_padding: ')[1].strip())
                elif 'temporal_padding:' in line:
                    parameters['temporal_padding'] = int(line.split('temporal_padding: ')[1].strip())
                
                # Parse other parameters
                elif 'index_path:' in line:
                    parameters['index_path'] = line.split('index_path: ')[1].strip()
                elif 'output_dir:' in line:
                    parameters['output_dir'] = line.split('output_dir: ')[1].strip()
                elif 'batch_size:' in line:
                    parameters['batch_size'] = int(line.split('batch_size: ')[1].strip())
                
                # Parse products
                elif 'Products to process:' in line:
                    products_str = line.split('Products to process: ')[1].strip()
                    parameters['products'] = eval(products_str)
        
        # Set default values for any missing parameters
        if 'index_path' not in parameters:
            parameters['index_path'] = "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/isas20_reference_index.pkl"
        
        # Set log directory based on output directory
        parameters['log_dir'] = os.path.join(parameters['output_dir'], 'logs')
        
        logger.info("Successfully parsed parameters from log file:")
        for key, value in parameters.items():
            if key != 'products':  # Don't print the full products dict
                logger.info(f"  {key}: {value}")
        
        return parameters
    
    except Exception as e:
        logger.error(f"Error parsing parameters from log file: {e}")
        raise

def parse_old_logs(log_dir, main_log_file):
    """
    Parse old log files to identify failed batches.
    
    Args:
        log_dir: Directory containing problematic station logs
        main_log_file: Path to the main log file
    
    Returns:
        list: List of tuples (batch_start, batch_end) for failed batches
    """
    logger = logging.getLogger(__name__)
    failed_batches = set()
    
    # Look for errors in the main log file
    try:
        with open(main_log_file, 'r') as f:
            for line in f:
                if 'ERROR' in line and 'Error processing batch' in line:
                    # Extract batch numbers from the error message
                    match = re.search(r'batch (\d+)-(\d+)', line)
                    if match:
                        batch_start = int(match.group(1))
                        batch_end = int(match.group(2))
                        failed_batches.add((batch_start, batch_end))
                        logger.info(f"Found failed batch in main log: {batch_start}-{batch_end}")
    except Exception as e:
        logger.warning(f"Error parsing main log file {main_log_file}: {e}")
    
    # Sort the batches by start index
    return sorted(failed_batches)

def process_individual_stations(parameters, failed_batches):
    """
    Process individual stations from failed batches using generate_satellite_data.py.
    """
    logger = logging.getLogger(__name__)
    
    # Create a temporary directory for individual station processing
    temp_dir = os.path.join(parameters['output_dir'], 'temp_individual_stations')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load and filter stations
    logger.info("Loading and filtering ISAS20 reference index...")
    all_stations = load_and_filter_index(
        parameters['index_path'],
        parameters['start_date'],
        parameters['end_date'],
        parameters['min_lat'],
        parameters['max_lat'],
        parameters['min_lon'],
        parameters['max_lon']
    )
    
    # Load the original index to get the full record format
    with open(parameters['index_path'], 'rb') as f:
        original_index = pickle.load(f)
    
    # Process each failed batch
    for batch_start, batch_end in failed_batches:
        logger.info(f"\nProcessing stations from failed batch {batch_start}-{batch_end}")
        
        # Get stations for this batch
        batch_stations = all_stations[batch_start:batch_end]
        
        # Create a temporary index file for this batch
        temp_index = os.path.join(temp_dir, f"temp_index_{batch_start:06d}-{batch_end:06d}.pkl")
        
        # Create a batch index in the correct format
        batch_index = {}
        for i, station in enumerate(batch_stations):
            station_idx = batch_start + i
            lat, lon, jd, file_path, profile_idx = station
            batch_index[(lat, lon, jd)] = {
                'file': file_path,
                'profile_index': profile_idx
            }
        
        with open(temp_index, 'wb') as f:
            pickle.dump(batch_index, f)
        
        # Build command for generate_satellite_data.py
        cmd = [
            "python", "generate_satellite_data.py",
            "--start_date", parameters['start_date'].strftime('%Y-%m-%d'),
            "--end_date", parameters['end_date'].strftime('%Y-%m-%d'),
            "--min_lat", str(parameters['min_lat']),
            "--max_lat", str(parameters['max_lat']),
            "--min_lon", str(parameters['min_lon']),
            "--max_lon", str(parameters['max_lon']),
            "--spatial_padding", str(parameters['spatial_padding']),
            "--temporal_padding", str(parameters['temporal_padding']),
            "--index_path", temp_index,
            "--output_dir", temp_dir,
            "--batch_size", str(parameters['batch_size'])
        ]
        
        # Create a log file for this batch
        batch_log_file = os.path.join(temp_dir, f"batch_{batch_start:06d}-{batch_end:06d}.log")
        
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            logger.info(f"Writing detailed output to: {batch_log_file}")
            
            # Run generate_satellite_data.py for this batch with output capture
            with open(batch_log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output to both the log file and our logger
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        log_f.write(output)
                        logger.info(output.strip())
                
                # Get any remaining output and the return code
                stdout, stderr = process.communicate()
                if stdout:
                    log_f.write(stdout)
                    logger.info(stdout.strip())
                if stderr:
                    log_f.write(stderr)
                    logger.error(stderr.strip())
                
                return_code = process.poll()
            
            # Check if the file was created and is valid
            output_file = os.path.join(
                temp_dir,
                f"satellite_data_batch_{batch_start:06d}-{batch_end:06d}.h5"
            )
            
            if os.path.exists(output_file):
                with h5py.File(output_file, 'r') as f:
                    # Check each product and variable
                    missing_products = []
                    for product in parameters['products']:
                        if product not in f:
                            missing_products.append(product)
                            continue
                        
                        # Check each variable in the product
                        for var in parameters['products'][product]:
                            if var not in f[product]:
                                missing_products.append(f"{product}/{var}")
                    
                    if not missing_products:
                        # Move the file to the final output directory
                        final_output = os.path.join(
                            parameters['output_dir'],
                            f"satellite_data_batch_{batch_start:06d}-{batch_end:06d}.h5"
                        )
                        
                        # Delete the old file if it exists
                        if os.path.exists(final_output):
                            logger.info(f"Deleting old file: {final_output}")
                            os.remove(final_output)
                        
                        # Move the new file to the final location
                        os.rename(output_file, final_output)
                        logger.info(f"Successfully processed batch {batch_start}-{batch_end}")
                    else:
                        logger.warning(f"Batch {batch_start}-{batch_end} failed: missing {', '.join(missing_products)}")
                        os.remove(output_file)
            else:
                logger.warning(f"Batch {batch_start}-{batch_end} failed: no output file created")
                if return_code != 0:
                    logger.error(f"Process exited with return code {return_code}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing batch {batch_start}-{batch_end}:")
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Output: {e.output}")
            logger.error(f"Error: {e.stderr}")
            if os.path.exists(output_file):
                os.remove(output_file)
        except Exception as e:
            logger.error(f"Unexpected error processing batch {batch_start}-{batch_end}:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            if os.path.exists(output_file):
                os.remove(output_file)
        finally:
            # Clean up temporary index file
            if os.path.exists(temp_index):
                os.remove(temp_index)

def delete_failed_batch_files(parameters, failed_batches):
    """Delete the HDF5 files corresponding to failed batches so that they can be re‑generated.

    Parameters
    ----------
    parameters : dict
        Parsed parameters dictionary produced by ``parse_parameters_from_log``.
    failed_batches : list[tuple[int, int]]
        List of (batch_start, batch_end) tuples for which the run failed.
    """
    logger = logging.getLogger(__name__)

    output_dir = parameters.get("output_dir", ".")
    if not failed_batches:
        logger.info("No failed batches to delete – nothing to do.")
        return

    logger.info("Deleting partial HDF5 files from previous failed batches …")
    for batch_start, batch_end in failed_batches:
        fname = os.path.join(
            output_dir,
            f"satellite_data_batch_{batch_start:06d}-{batch_end:06d}.h5"
        )
        if os.path.exists(fname):
            try:
                os.remove(fname)
                logger.info(f"  ✓ Deleted {fname}")
            except Exception as e:
                logger.warning(f"  ✗ Could not delete {fname}: {e}")
        else:
            logger.debug(f"  (File not found – already gone) {fname}")

def main():
    parser = argparse.ArgumentParser(
        description="Resume processing of failed batches from a previous run. "
                    "Use --delete_only to just wipe the failed batch files so that "
                    "they will be regenerated on the next call to generate_satellite_data.py."
    )
    
    # Only argument needed is the log file path
    parser.add_argument('log_file', type=str,
                        help="Path to the log file from the previous run")
    parser.add_argument("-d", "--delete_only", action="store_true",
                        help="Only delete failed batch files and exit (do not re‑process).")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(os.path.dirname(args.log_file))
    
    # Parse parameters from log file
    logger.info(f"Parsing parameters from log file: {args.log_file}")
    parameters = parse_parameters_from_log(args.log_file)
    
    # Parse old logs to find failed batches
    logger.info("Parsing old log files…")
    failed_batches = parse_old_logs(parameters['log_dir'], args.log_file)
    
    if not failed_batches:
        logger.info("No failed batches found in logs.")
        return

    # NEW: delete only mode
    if args.delete_only:
        delete_failed_batch_files(parameters, failed_batches)
        logger.info("Deletion step completed. Rerun generate_satellite_data.py with the same"
                    " parameters; it will rebuild the deleted batches.")
        logger.info("Suggested command:")
        logger.info("python generate_satellite_data.py \\")
        logger.info("    --start_date %s --end_date %s \\" % (parameters['start_date'].strftime('%Y-%m-%d'), parameters['end_date'].strftime('%Y-%m-%d')))
        logger.info("    --min_lat %s --max_lat %s \\" % (parameters['min_lat'], parameters['max_lat']))
        logger.info("    --min_lon %s --max_lon %s \\" % (parameters['min_lon'], parameters['max_lon']))
        logger.info("    --spatial_padding %s --temporal_padding %s \\" % (parameters['spatial_padding'], parameters['temporal_padding']))
        logger.info("    --index_path %s \\" % parameters['index_path'])
        logger.info("    --output_dir %s \\" % parameters['output_dir'])
        logger.info("    --batch_size %s \\" % parameters['batch_size'])
        logger.info("    --missing_batches_only")
        return
    
    logger.info(f"Found {len(failed_batches)} failed batches to reprocess")
    for batch_start, batch_end in failed_batches:
        logger.info(f"  Batch {batch_start}-{batch_end}")
    
    # Process failed batches
    process_individual_stations(parameters, failed_batches)
    
    logger.info("Resume processing completed")

if __name__ == "__main__":
    main() 