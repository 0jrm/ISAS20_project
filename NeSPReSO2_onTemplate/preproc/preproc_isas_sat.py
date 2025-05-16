import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta
import pickle
import sys
import h5py
import json
import os
from typing import Dict, Tuple, Optional, Union

# Given IO specifications, load, preprocess, and save the data and it's masks

# Inputs:
# - data_path (str): File path to the dataset.
# - BBox (tuple): Bounding box for the data (min_lat, max_lat, min_lon, max_lon).
# - input_params (dict): Parameters to include as input. (time, geo, sst, sss, adt, satSSH, satSST, windspeed, etc)
# - input_processing (dict): How inputs should be processed. (None, norm, pca, vae, DIRESA, etc)
# - spatial_step (int): Desired spatial resolution for the sat data, how many 0.25 degree steps to take.
# - spatial_pad (int): Desired spatial pad size for the sat data (0=1x1, 1=3x3, 2=5x5, etc).
# - temporal_pad (int): Desired temporal padding for the sat data (how many previous days to include).
# - output_params (dict): Parameters to include as output.
# - output_processing (dict): How outputs should be processed. (None, norm, pca, vae, DIRESA, etc)
# - save_path (str): File path to save the data and it's masks.

# Outputs:
# - saves the tensor data and it's masks in a pickle file

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from a JSON file or return default configuration.

    Parameters:
        config_path (str, optional): Path to the configuration JSON file.

    Returns:
        dict: Configuration dictionary.
    """
    if config_path is None:
        # Default configuration
        return {
            "data_path": "/unity/g2/jmiranda/SubsurfaceFields/Data/ISAS20_ARGO/ISAS20_project/data/NeSPReSO_v2_GoM_sat",
            "BBox": None,
            "input_params": {
                "bathymetry": ["elevation"],
                "ssh": ["adt", "sla", "ugos", "vgos"],
                "ostia": ["analysed_sst"],
                "sss": ["sos"],
                "wind": ["windspeed", "u_wind", "v_wind"]
            },
            "input_processing": "norm",
            "spatial_step": 1,
            "spatial_pad": 2,
            "temporal_pad": 3,
            "output_params": {
                "temperature": ["TEMP"],
                "salinity": ["PSAL"]
            },
            "output_processing": "norm"
        }
    
    with open(config_path, 'r') as f:
        return json.load(f)

def normalize_ignore_nan(arr, axis=None):
    """
    Normalize an array by subtracting the mean and dividing by the standard deviation, ignoring NaN values.
    NaN values are replaced with 0 after normalization.

    Parameters:
        arr (numpy.ndarray): Input array to normalize.
        axis (int or tuple, optional): Axis or axes along which to normalize. If None, normalize across all dimensions.

    Returns:
        tuple: (normalized_array, mean, std)
            - normalized_array: Normalized array with NaNs replaced by 0
            - mean: Mean values used for normalization
            - std: Standard deviation values used for normalization
    """
    # Count non-NaN values along the specified axis
    if axis is not None:
        n_valid = np.sum(~np.isnan(arr), axis=axis, keepdims=True)
    else:
        n_valid = np.sum(~np.isnan(arr))

    # Calculate mean only where we have valid values
    mean = np.nanmean(arr, axis=axis, keepdims=True)
    mean = np.where(n_valid > 0, mean, 0)  # Replace NaN means with 0

    # Calculate std only where we have enough valid values (at least 2 for meaningful std)
    std = np.nanstd(arr, axis=axis, keepdims=True)
    std = np.where(n_valid > 1, std, 1)  # Replace invalid stds with 1

    # Normalize
    normed = (arr - mean) / std
    normed = np.where(np.isnan(normed), 0, normed)
    
    return normed, mean, std


def get_bbox_mask(lat, lon, bbox):
    """
    Create a boolean mask for stations within a bounding box.

    Parameters:
        lat (numpy.ndarray): Array of latitudes.
        lon (numpy.ndarray): Array of longitudes.
        bbox (tuple or None): Bounding box as (min_lat, max_lat, min_lon, max_lon). If None, returns all True.

    Returns:
        numpy.ndarray: Boolean mask indicating stations within the bounding box.
    """
    if bbox is None:
        return np.ones_like(lat, dtype=bool)
    min_lat, max_lat, min_lon, max_lon = bbox
    return (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)


def reshape_time(data, temporal_pad):
    """
    Adjust data for a desired temporal padding.

    Parameters:
        data (numpy.ndarray): Input data with shape (N, T, ...).
        temporal_pad (int): Number of time steps to stack.

    Returns:
        numpy.ndarray: Sliced data with shape (N, temporal_pad+1, ...).
    """
    N, T = data.shape[:2]
    try:
        return data[:, :temporal_pad+1, ...]
    except:
        print(f"Error: data contains {T} time steps and temporal_pad must be smaller than this, currently {temporal_pad}.")
        raise

def preprocess_data(config: Dict) -> Dict:
    """
    Preprocess satellite and profile data according to the provided configuration.

    Parameters:
        config (dict): Configuration dictionary containing all preprocessing parameters.

    Returns:
        dict: Processed data dictionary containing inputs, outputs, masks, and statistics.
    """
    data_path = config["data_path"]
    sat_file = os.path.join(data_path, 'satellite_NeSPReSO_v2_GoM.h5')
    prof_file = os.path.join(data_path, 'profiles_NeSPReSO_v2_GoM.h5')

    # Load station positions from profiles
    with h5py.File(prof_file, 'r') as pf:
        station_lat = pf['model']['LATITUDE'][:]
        station_lon = pf['model']['LONGITUDE'][:]
        bbox_mask = get_bbox_mask(station_lat, station_lon, config["BBox"])
        station_indices = np.where(bbox_mask)[0]

    # Prepare output dict
    processed = {'inputs': {}, 'outputs': {}, 'masks': {}, 'station_indices': station_indices, 'stats': {}}

    # Load and preprocess input variables
    with h5py.File(sat_file, 'r') as sf:
        for group, vars in config["input_params"].items():
            g = sf[group]
            for var in vars:
                d = g[var][:]  # shape: (N, ...) or (N, T, ...)
                # Select only stations in bbox
                d = d[station_indices]
                mask = np.isnan(d)
                if config["input_processing"] == 'norm':
                    d, input_mean, input_std = normalize_ignore_nan(d)
                    processed['stats'][f'{group}_{var}_mean'] = input_mean
                    processed['stats'][f'{group}_{var}_std'] = input_std
                # Stack temporal dimension if needed
                if d.ndim >= 3 and d.shape[1] >= config["temporal_pad"]:
                    d = reshape_time(d, config["temporal_pad"])
                    mask = reshape_time(mask, config["temporal_pad"])
                processed['inputs'][f'{group}_{var}'] = d
                processed['masks'][f'{group}_{var}'] = mask

    # Load and preprocess output variables
    with h5py.File(prof_file, 'r') as pf:
        for group, vars in config["output_params"].items():
            g = pf['model']
            for var in vars:
                d = g[var][:]
                d = d[station_indices]
                mask = np.isnan(d)
                if config["output_processing"] == 'norm':
                    d, output_mean, output_std = normalize_ignore_nan(d, axis=0) # normalize across depths
                    processed['stats'][f'{group}_{var}_mean'] = output_mean
                    processed['stats'][f'{group}_{var}_std'] = output_std
                processed['outputs'][f'{group}_{var}'] = d
                processed['masks'][f'{group}_{var}'] = mask

    return processed

def save_processed_data(processed: Dict, save_path: str) -> None:
    """
    Save processed data to a pickle file.

    Parameters:
        processed (dict): Processed data dictionary.
        save_path (str): Path where to save the pickle file.
    """
    with open(save_path, 'wb') as f:
        pickle.dump(processed, f)
    print(f"Preprocessed data saved to {save_path}")

def __main__(config_path: Optional[str] = None):
    """
    Main function to run the preprocessing pipeline.

    Parameters:
        config_path (str, optional): Path to the configuration JSON file. If None, uses default configuration.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set save path
    save_path = os.path.join(config["data_path"], 'preprocessed_satellite_data.pkl')
    
    # Preprocess data
    processed = preprocess_data(config)
    
    # Save processed data
    save_processed_data(processed, save_path)

if __name__ == "__main__":
    # Check if running in interactive mode (IPython/Jupyter)
    try:
        # This will raise NameError if not in IPython/Jupyter
        get_ipython()
        # If we get here, we're in IPython/Jupyter
        __main__(None)  # Use default configuration
    except NameError:
        # We're running from command line
        config_path = sys.argv[1] if len(sys.argv) > 1 else None
        __main__(config_path)    




