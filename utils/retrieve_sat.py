"""
Satellite Data Retrieval Module

This module provides functions to retrieve and interpolate satellite data for oceanographic applications.
The main function is retrieve_satellite_data() which processes multiple queries in batch.

###
retrieve_satellite_data(queries: list, products: dict, spatial_pad: int, temporal_pad: int = 0) -> dict:
###

# Retrieve and interpolate satellite data for a batch of queries.

Each query is a tuple (latitude, longitude, julian_date) where julian_date is in
astronomical Julian days. Singleton dimensions are removed (outputs are 3D at most).
For static products (e.g. bathymetry), the extra time dimension is dropped.

Parameters:
    queries (list): List of (lat, lon, julian_date) tuples.
    products (dict): Mapping of product names to lists of variable names.
    spatial_pad (int): Number of grid steps for spatial padding (gridsize = 2*spatial_pad + 1, e.g., 1 for 3x3, 16 for 33x33).
    temporal_pad (int, optional): Number of previous days to include for daily products. If 0, products are 2D, otherwise 3D (time[0=query_date], lat, lon).
    
Returns:
    dict: Mapping of query index to a dict of product results. 
        Please see the example usage at the bottom of the file for more details.
        Each product result is a dict:
            { 
            "file": file_info, 
            "data": { var: interpolated_array },
            "coordinates": {
                "latitude": array of latitudes,
                "longitude": array of longitudes
            }
            }.
"""

import os
import re
import glob
import numpy as np
import xarray as xr
import logging
import functools
from datetime import datetime, timedelta
from astropy.time import Time
from pathlib import Path
import warnings
from tqdm import tqdm
import h5py

# Suppress common warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in reduce")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# =============================================================================
# Logging Configuration
# =============================================================================
# Configure logging with minimal output for performance
logging_level = logging.WARNING
logger = logging.getLogger('retrieve_sat')
logger.setLevel(logging_level)
# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Add a simple handler with minimal formatting
handler = logging.StreamHandler()
handler.setLevel(logging_level)
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# Disable propagation to root logger to avoid duplicate messages
logger.propagate = False

# =============================================================================
# Global Configuration
# =============================================================================
# Root directories for satellite products
ROOT_DIRS = {
    "bathymetry": "/unity/g2/jmiranda/SubsurfaceFields/Data/Bathymetry",
    "ostia":      "/unity/g2/jmiranda/SubsurfaceFields/Data/OISST/OSTIA",
    "remss":      "/unity/g2/jmiranda/SubsurfaceFields/Data/OISST/REMSS",
    "wind":       "/unity/g2/jmiranda/SubsurfaceFields/Data/Wind",
    "ssh":        "/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSH",
    "sss":        "/unity/g2/jmiranda/SubsurfaceFields/Data/CMEMS/SSS"
}

# Product metadata
PRODUCT_METADATA = {
    "bathymetry": {
        "is_static": True,
        "regex": None,
        "date_fmt": None,
        "var_mapping": {"elevation": "elevation"}
    },
    "ostia": {
        "is_static": False,
        "regex": r"(\d{14})-.*OSTIA.*\.nc",
        "date_fmt": "%Y%m%d%H%M%S",
        "var_mapping": {"analysed_sst": "analysed_sst"}
    },
    "remss": {
        "is_static": False,
        "regex": r"(\d{14})-.*REMSS.*\.nc",
        "date_fmt": "%Y%m%d%H%M%S",
        "var_mapping": {"SST": ["analysed_sst", "SST"]}
    },
    "wind": {
        "is_static": False,
        "regex": r".*daily_(\d{8})\.nc",
        "date_fmt": "%Y%m%d",
        "var_mapping": {
            "u_wind": "u_wind", 
            "v_wind": "v_wind", 
            "windspeed": "windspeed"
        }
    },
    "ssh": {
        "is_static": False,
        "regex": r"SSH_(\d{4})\.nc",
        "date_fmt": "%Y",
        "var_mapping": {
            "adt": "adt", 
            "sla": "sla", 
            "ugos": "ugos", 
            "vgos": "vgos"
        }
    },
    "sss": {
        "is_static": False,
        "regex": r"SSS_(\d{8})\.nc",
        "date_fmt": "%Y%m%d",
        "var_mapping": {"sos": "sos"}
    }
}

# =============================================================================
# File lookup and coordinate helper functions
# =============================================================================
@functools.lru_cache(maxsize=None)
def get_product_files(product):
    """
    Get all available files for a product and cache the result.
    
    Parameters:
        product (str): The product name
        
    Returns:
        list: List of file paths
    """
    folder = ROOT_DIRS.get(product)
    if folder is None:
        return []
    
    file_pattern = os.path.join(folder, "*.nc")
    return glob.glob(file_pattern)

@functools.lru_cache(maxsize=1024)
def select_candidate_file(product: str, query_dt: datetime) -> str:
    """
    Select the most appropriate NetCDF file for a given product and query date.
    Results are cached for performance.
    
    Parameters:
        product (str): The product name
        query_dt (datetime): The query date
    
    Returns:
        str: Full path to the candidate file, or None if no file is found
    """
    files = get_product_files(product)
    if not files:
        return None
    
    metadata = PRODUCT_METADATA.get(product)
    if metadata is None:
        logger.warning(f"No metadata defined for product {product}")
        return None
    
    # For static products, return the first file
    if metadata["is_static"]:
        return files[0]
    
    regex, date_fmt = metadata["regex"], metadata["date_fmt"]
    if regex is None or date_fmt is None:
        logger.warning(f"Missing regex or date format for product {product}")
        return None
    
    candidate, min_diff = None, None
    for f in files:
        m = re.search(regex, os.path.basename(f), re.IGNORECASE)
        if m:
            date_str = m.group(1)
            try:
                file_dt = datetime.strptime(date_str, date_fmt)
            except Exception as e:
                logger.debug(f"Error parsing date from {f}: {e}")
                continue
            diff = abs((query_dt - file_dt).total_seconds())
            if min_diff is None or diff < min_diff:
                min_diff, candidate = diff, f
    
    return candidate

def get_coord_names(ds: xr.Dataset) -> tuple:
    """
    Returns the coordinate names for latitude, longitude, and time.
    
    Parameters:
        ds (xr.Dataset): The dataset
        
    Returns:
        tuple: (lat_name, lon_name, time_name)
    """
    lat_name = next((name for name in ["lat", "latitude"] if name in ds.coords), None)
    lon_name = next((name for name in ["lon", "longitude"] if name in ds.coords), None)
    time_name = "time" if "time" in ds.coords else None
    return lat_name, lon_name, time_name

def get_variable_name(product, var, ds):
    """
    Get the actual variable name in the dataset based on product metadata.
    
    Parameters:
        product (str): The product name
        var (str): The requested variable
        ds (xr.Dataset): The dataset
        
    Returns:
        str: The actual variable name in the dataset, or None if not found
    """
    metadata = PRODUCT_METADATA.get(product)
    if metadata is None:
        return var
    
    var_mapping = metadata.get("var_mapping", {})
    mapped_var = var_mapping.get(var, var)
    
    # If mapping is a list, try each option
    if isinstance(mapped_var, list):
        for option in mapped_var:
            if option in ds.variables:
                return option
        return None
    
    return mapped_var if mapped_var in ds.variables else None

# =============================================================================
# Interpolation functions
# =============================================================================
def normalize_longitude(lon):
    """Normalize longitude to -180 to 180 range."""
    lon = lon % 360
    if lon > 180:
        lon -= 360
    return lon

def build_target_coordinates(ds, query_lat, query_lon, spatial_pad):
    """
    Build target coordinates for interpolation.
    
    Parameters:
        ds (xr.Dataset): The dataset
        query_lat (float): Query latitude
        query_lon (float): Query longitude
        spatial_pad (int): Spatial padding size
        
    Returns:
        tuple: (target_lats, target_lons)
    """
    lat_name, lon_name, _ = get_coord_names(ds)
    lon_vals = ds[lon_name].values
    
    # Handle the case when spatial_pad is 0
    if spatial_pad == 0:
        # Return single coordinates
        query_lon_norm = normalize_longitude(query_lon)
        # If dataset uses 0-360 range, convert target longitude to match
        if lon_vals.min() >= 0 and lon_vals.max() > 180 and query_lon_norm < 0:
            query_lon_norm += 360
        return np.array([query_lat]), np.array([query_lon_norm])
    
    # For non-zero padding, continue with existing logic
    # Enforce fixed spatial resolution of 0.25 degrees
    lat_res = 0.25
    lon_res = 0.25
    
    # Create target coordinates using fixed resolution
    target_lats = query_lat + np.arange(-spatial_pad, spatial_pad+1) * lat_res
    
    # Handle longitude consistently in -180 to 180 range
    query_lon_norm = normalize_longitude(query_lon)
    target_lons = np.array([normalize_longitude(query_lon_norm + i * lon_res) 
                           for i in range(-spatial_pad, spatial_pad+1)])
    
    # If dataset uses 0-360 range, convert target longitudes to match
    if lon_vals.min() >= 0 and lon_vals.max() > 180:
        target_lons = np.where(target_lons < 0, target_lons + 360, target_lons)
    
    return target_lats, target_lons

def build_multitime_interp_kwargs(ds: xr.Dataset, query_lat: float, query_lon: float,
                                  query_dt: datetime, spatial_pad: int, temporal_pad: int) -> dict:
    """
    Build keyword arguments for xarray.interp() for multi-time datasets.
    Computes the target spatial grid and selects the nearest (or a slice of) time steps.
    Time values are ordered with most recent first.
    """
    lat_name, lon_name, time_name = get_coord_names(ds)
    target_lats, target_lons = build_target_coordinates(ds, query_lat, query_lon, spatial_pad)
    
    # Handle time dimension
    if time_name is not None:
        time_vals = ds[time_name].values
        query_time_np = np.datetime64(query_dt)
        time_diff = np.abs(time_vals - query_time_np)
        nearest_idx = int(time_diff.argmin())
        
        if temporal_pad > 0:
            start_idx = max(0, nearest_idx - temporal_pad)
            # Changed: Reverse the time slice to have most recent first
            target_times = time_vals[start_idx:nearest_idx+1][::-1]
        else:
            target_times = time_vals[nearest_idx:nearest_idx+1]
    else:
        target_times = None
    
    # Build interpolation kwargs
    interp_kwargs = {lat_name: target_lats, lon_name: target_lons}
    if target_times is not None:
        interp_kwargs[time_name] = target_times
    
    # Add depth dimension if present
    if "zlev" in ds.coords:
        interp_kwargs["zlev"] = ds["zlev"].values
    
    return interp_kwargs

def safe_interpolate(ds, var_key, interp_kwargs, method="nearest"):
    try:
        # Minimize logging for performance
        input_data = ds[var_key]
        logger.info(f"Interpolating {var_key}")
        
        # Get all dimension names
        lat_name, lon_name, time_name = get_coord_names(ds)
        
        # Handle depth/zlev dimension if present
        depth_name = None
        if "depth" in input_data.dims:
            depth_name = "depth"
        elif "zlev" in input_data.dims:
            depth_name = "zlev"
        
        # Create list of all dimensions in the correct order
        dims = []
        if time_name in input_data.dims:
            dims.append(time_name)
        if depth_name is not None:
            dims.append(depth_name)
        dims.extend([lat_name, lon_name])
        
        # Ensure all dimensions are present in the data
        missing_dims = [dim for dim in dims if dim not in input_data.dims]
        if missing_dims:
            logger.warning(f"Missing dimensions {missing_dims} in data, skipping")
            return None
        
        # Transpose to the correct order
        input_data = input_data.transpose(*dims)
        
        # Check if we're doing single-point interpolation (spatial_pad=0)
        is_single_point = (len(interp_kwargs[lat_name]) == 1 and len(interp_kwargs[lon_name]) == 1)
        
        # Performance optimization: For nearest neighbor interpolation or single points,
        # use sel with method='nearest' which is faster than interp
        if method == "nearest" or is_single_point:
            # Convert interp_kwargs to selection kwargs
            sel_kwargs = {}
            for coord, values in interp_kwargs.items():
                if isinstance(values, np.ndarray):
                    sel_kwargs[coord] = values[0] if len(values) == 1 else values
                else:
                    sel_kwargs[coord] = values
            result = input_data.sel(method='nearest', **sel_kwargs)
        else:
            result = input_data.interp(method=method, **interp_kwargs)
        
        # Squeeze out singleton dimensions
        result = result.squeeze()
        
        # For single-point interpolation, ensure we maintain the correct dimensions
        if is_single_point:
            # Add back lat/lon dimensions if they were squeezed out
            if lat_name not in result.dims and lat_name in interp_kwargs:
                result = result.expand_dims(lat_name)
            if lon_name not in result.dims and lon_name in interp_kwargs:
                result = result.expand_dims(lon_name)
        
        # Ensure the result has the correct shape (time, lat, lon) or (lat, lon) for static products
        if time_name in result.dims:
            # For time-varying products, ensure shape is (time, lat, lon)
            result = result.transpose(time_name, lat_name, lon_name)
        else:
            # For static products, ensure shape is (lat, lon)
            result = result.transpose(lat_name, lon_name)
        
        return result
    except Exception as e:
        logger.error(f"Error interpolating {var_key}: {str(e)}")
        return None

def process_daily_product(product: str, query_lat: float, query_lon: float, query_dt: datetime,
                          spatial_pad: int, temporal_pad: int, products: dict) -> dict:
    """
    For daily datasets (one time stamp per file), gather files for the query day and preceding days
    (if temporal_pad > 0), extract the spatially interpolated data from each file, and stack them along
    a new time axis. The time index 0 corresponds to the query date, with subsequent indices
    representing previous days.
    """
    logger.info(f"\n=== Processing Query ===")
    logger.info(f"Product: {product}")
    logger.info(f"Query coordinates: ({query_lat}, {query_lon})")
    logger.info(f"Query date: {query_dt}")
    
    # Skip temporal padding for static products
    if PRODUCT_METADATA.get(product, {}).get("is_static", False):
        temporal_pad = 0
    
    # Gather files for each day
    file_entries = []
    missing_files = []
    for offset in range(temporal_pad + 1):
        dt_i = query_dt - timedelta(days=offset)
        fp = select_candidate_file(product, dt_i)
        if fp:
            file_entries.append((offset, fp))
            logger.info(f"Found file for offset {offset}: {os.path.basename(fp)}")
        else:
            missing_files.append(dt_i)
            logger.warning(f"No file found for {product} at date offset {offset} (date {dt_i}).")
    
    if not file_entries:
        logger.error(f"No files found for product {product} around {query_dt}")
        return None
    
    # Adjust spatial padding for SSS to account for higher resolution
    effective_spatial_pad = spatial_pad * 2 if product == "sss" else spatial_pad
    
    # Sort by offset (most recent first) to ensure time index 0 is query date
    file_entries = sorted(file_entries, key=lambda x: x[0], reverse=True)
    stacked_vars = {}
    
    # Initialize expected shape based on first file
    expected_shape = None
    
    # Process each file
    for offset, file_path in file_entries:
        try:
            with xr.open_dataset(file_path, decode_times=True) as ds:
                lat_name, lon_name, time_name = get_coord_names(ds)
                target_lats, target_lons = build_target_coordinates(ds, query_lat, query_lon, effective_spatial_pad)
                
                # For SSS, subsample the points to match other products' resolution
                if product == "sss":
                    target_lats = target_lats[::2]
                    target_lons = target_lons[::2]
                
                interp_kw = {lat_name: target_lats, lon_name: target_lons}
                
                # Add time dimension if present
                if time_name is not None:
                    # For SSH, select only the time slice closest to query_dt
                    if product == "ssh":
                        time_vals = ds[time_name].values
                        time_diff = np.abs(time_vals - np.datetime64(query_dt))
                        nearest_idx = int(time_diff.argmin())
                        interp_kw[time_name] = time_vals[nearest_idx]
                    else:
                        # For other products, use the first time in the file
                        interp_kw[time_name] = ds[time_name].values[0]
                
                # Process each variable
                for var in products[product]:
                    var_key = get_variable_name(product, var, ds)
                    if var_key is None:
                        logger.warning(f"Variable '{var}' not found in {file_path}")
                        continue
                    
                    extracted = safe_interpolate(ds, var_key, interp_kw)
                    if extracted is not None:
                        data = extracted.values
                        
                        # Set expected shape from first successful extraction
                        if expected_shape is None:
                            expected_shape = data.shape
                        
                        # Ensure consistent shape
                        if data.shape != expected_shape:
                            logger.warning(f"Inconsistent shape for {var} in {file_path}: {data.shape} vs {expected_shape}")
                            # Create nan array with expected shape
                            data = np.full(expected_shape, np.nan)
                        
                        stacked_vars.setdefault(var, []).append(data)
                    else:
                        # If extraction failed, add nan array with expected shape
                        if expected_shape is not None:
                            stacked_vars.setdefault(var, []).append(np.full(expected_shape, np.nan))
                        logger.warning(f"Failed to extract {var_key} data")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Stack results
    stacked_results = {}
    for var, arr_list in stacked_vars.items():
        if not arr_list:
            logger.warning(f"No valid data to stack for variable {var}")
            continue
        try:
            # Stack arrays along time dimension
            stacked_results[var] = np.stack(arr_list, axis=0)
            
            # Log statistics of stacked data
            nan_percentage = np.isnan(stacked_results[var]).mean() * 100
            logger.info(f"Stacked {var} statistics:")
            logger.info(f"  NaN percentage: {nan_percentage:.2f}%")
            if not np.all(np.isnan(stacked_results[var])):
                logger.info(f"  Range: [{np.nanmin(stacked_results[var])}, {np.nanmax(stacked_results[var])}]")
        except Exception as e:
            logger.error(f"Error stacking variable '{var}': {str(e)}")
    
    # Add time information and missing files info
    if not PRODUCT_METADATA.get(product, {}).get("is_static", False):
        # Create time array with query_dt at index 0 and previous days
        times = np.array([query_dt - timedelta(days=i) for i in range(temporal_pad + 1)])
        stacked_results["time"] = times
    if missing_files:
        stacked_results["missing_files"] = missing_files
    
    return stacked_results

# =============================================================================
# Batch processing function
# =============================================================================
def retrieve_satellite_data(queries: list, products: dict, spatial_pad: int, temporal_pad: int = 0, max_batch_size: int = None) -> dict:
    """
    Retrieve and interpolate satellite data for a batch of queries.
    
    Each query is a tuple (latitude, longitude, julian_date) where julian_date is in
    astronomical Julian days. For each product, if temporal_pad > 0, data will be gathered
    from multiple files spanning the temporal padding window. Singleton dimensions are removed
    using np.squeeze(). For static products (e.g. bathymetry), the extra time dimension is dropped.
    
    Parameters:
        queries (list): List of (lat, lon, julian_date) tuples.
        products (dict): Mapping of product names to lists of variable names.
        spatial_pad (int): Number of grid steps for spatial padding (e.g., 1 for 3x3).
        temporal_pad (int, optional): Number of previous days to include for daily products.
        max_batch_size (int, optional): Maximum number of queries to process in a single batch.
            If provided, queries will be split into smaller batches of this size.
        
    Returns:
        dict: Mapping of query index to a dict of product results. Each product result is a dict:
              { 
                "file": file_info, 
                "data": { var: interpolated_array },
                "coordinates": {
                    "latitude": array of latitudes,
                    "longitude": array of longitudes
                }
              }.
    """
    # If max_batch_size is provided and less than the number of queries, split into batches
    if max_batch_size and len(queries) > max_batch_size:
        logger.info(f"Splitting {len(queries)} queries into batches of {max_batch_size}")
        results = {}
        for i in range(0, len(queries), max_batch_size):
            batch_queries = queries[i:i + max_batch_size]
            batch_results = _process_batch(batch_queries, products, spatial_pad, temporal_pad)
            results.update(batch_results)
        return results
    else:
        return _process_batch(queries, products, spatial_pad, temporal_pad)

def _process_batch(queries: list, products: dict, spatial_pad: int, temporal_pad: int = 0) -> dict:
    """Internal function to process a single batch of queries."""
    results = {}
    
    # Add progress bar for query processing
    for idx, (qlat, qlon, qjd) in enumerate(tqdm(queries, desc="Retrieving satellite data", unit="profile")):
        results[idx] = {}
        
        # Convert Julian date to datetime
        try:
            qdt = Time(qjd, format="jd").to_datetime()
        except Exception as e:
            logger.warning(f"Error converting Julian date {qjd}: {e}")
            continue
        
        # Process each product
        for prod in products.keys():
            # Skip if product not in ROOT_DIRS
            if prod not in ROOT_DIRS:
                logger.warning(f"Product {prod} not found in ROOT_DIRS")
                continue
            
            # Get candidate file for the query date
            candidate_file = select_candidate_file(prod, qdt)
            if candidate_file is None:
                logger.debug(f"No file found for product {prod} on {qdt}")
                continue
            
            # Check if product is static
            is_static = PRODUCT_METADATA.get(prod, {}).get("is_static", False)
            
            # For static products, process as single-time product
            if is_static:
                try:
                    with xr.open_dataset(candidate_file, decode_times=True) as ds:
                        lat_name, lon_name, time_name = get_coord_names(ds)
                        target_lats, target_lons = build_target_coordinates(ds, qlat, qlon, spatial_pad)
                        
                        interp_kw = {
                            lat_name: target_lats,
                            lon_name: target_lons
                        }
                        
                        # Add depth dimension if present
                        if "zlev" in ds.coords:
                            interp_kw["zlev"] = ds["zlev"].values
                        
                        prod_result = {
                            "file": candidate_file, 
                            "data": {},
                            "coordinates": {
                                "latitude": target_lats,
                                "longitude": target_lons
                            }
                        }
                        
                        for var in products[prod]:
                            var_key = get_variable_name(prod, var, ds)
                            if var_key is None:
                                logger.debug(f"Variable '{var}' not found in {candidate_file}")
                                continue
                            
                            extracted = safe_interpolate(ds, var_key, interp_kw)
                            if extracted is None:
                                continue
                            
                            data_out = np.squeeze(extracted.values)
                            prod_result["data"][var] = data_out
                            
                        results[idx][prod] = prod_result
                        
                except Exception as e:
                    logger.warning(f"Error processing static product {prod}: {e}")
                continue
            
            # For non-static products, use process_daily_product to handle temporal padding
            # This will gather data from multiple files if needed
            daily_data = process_daily_product(prod, qlat, qlon, qdt, spatial_pad, temporal_pad, products)
            if daily_data is not None:
                # Get coordinates from the candidate file
                with xr.open_dataset(candidate_file, decode_times=True) as ds:
                    target_lats, target_lons = build_target_coordinates(ds, qlat, qlon, spatial_pad)
                    
                    prod_result = {
                        "file": f"Files from {qdt - timedelta(days=temporal_pad)} to {qdt}",
                        "data": daily_data,
                        "coordinates": {
                            "latitude": target_lats,
                            "longitude": target_lons
                        }
                    }
                    results[idx][prod] = prod_result
    
    return results

# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # Example query list: (latitude, longitude, julian_date)
    queries = [
        # (45.0, -30.0, 2459020.5),
        # (45.5, -29.5, 2459020.5),
        # (44.8, -30.2, 2459020.5),
        # (45.2, -29.8, 2459025.5),
        # (45.1, -30.1, 2459025.5)
        (23.11, -96.74, 2457012.00),
        (27.81, -94.53, 2457013.00),
        (25.50, -94.05, 2457013.00),
        (24.72, -91.40, 2457014.00)
    ]

    # Define the satellite products and the variables to extract
    products = {
        "bathymetry": ["elevation"],
        "ostia": ["analysed_sst"],
        "sss": ["sos"],
        # "remss": ["SST"],
        "wind": ["windspeed", "u_wind", "v_wind"],
        "ssh": ["adt", "sla", "ugos", "vgos"]
    }

    spatial_padding = 16  # e.g., extract a 3x3 region
    temporal_padding = 6  # Include the query day plus one previous day for daily products

    # Set logging level for testing
    logging.basicConfig(level=logging.INFO)

    # Retrieve data
    results = retrieve_satellite_data(queries, products, spatial_padding, temporal_padding)

    # Print results summary
    for idx, res in results.items():
        print(f"\nQuery {idx}: {queries[idx]}")
        for prod, info in res.items():
            print(f"  Product: {prod}")
            print(f"    File info: {info['file']}")
            for var, data in info["data"].items():
                if var != "time":
                    print(f"    Variable '{var}': data shape = {np.shape(data)}")
                    
    # #print data from the first query the same location and all times, for all products
    print(results[0]["ostia"]["data"]["analysed_sst"][:,0,0])
    print(results[0]["sss"]["data"]["sos"][:,0,0])
    print(results[0]["wind"]["data"]["windspeed"][:,0,0])
    print(results[0]["ssh"]["data"]["adt"][:,0,0])