from typing import List, Dict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

class SatelliteFetcher:
    def __init__(self, cache_dir: str, max_workers: int = 4, max_batch_size: int = 100):
        """
        Initialize the SatelliteFetcher with cache directory and configuration.
        
        Args:
            cache_dir (str): Directory to store cached satellite data
            max_workers (int): Maximum number of worker threads
            max_batch_size (int): Maximum number of stations to process in a single batch
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.max_batch_size = max_batch_size
        self.cache_file = self.cache_dir / "satellite_cache.h5"
        self.cache = {}
        self._load_cache()
        
        # Define satellite products to fetch
        self.products = {
            "bathymetry": ["elevation"],
            "ostia": ["analysed_sst"],
            "sss": ["sos"],
            "wind": ["windspeed", "u_wind", "v_wind"],
            "ssh": ["adt", "sla", "ugos", "vgos"]
        }
        
        # Reference date for Julian day conversion (1950-01-01)
        self.ref_date = datetime(1950, 1, 1)
    
    def fetch_batch(self, stations: List[Dict]) -> List[Dict]:
        """
        Fetch satellite data for a batch of stations.
        
        Args:
            stations (List[Dict]): List of station dictionaries containing lat, lon, and timestamp
            
        Returns:
            List[Dict]: List of station dictionaries with added satellite data
        """
        # Initialize results
        cached_results = {}
        queries_to_fetch = []
        station_indices = []
        
        # Check cache for each station
        for i, station in enumerate(stations):
            cache_key = self._get_cache_key(
                station['latitude'],
                station['longitude'],
                station['timestamp']
            )
            
            if cache_key in self.cache:
                cached_results[i] = self.cache[cache_key]
            else:
                queries_to_fetch.append((
                    station['latitude'],
                    station['longitude'],
                    station['timestamp']
                ))
                station_indices.append(i)
        
        # If we have queries to fetch, do them all at once
        if queries_to_fetch:
            logger.info(f"Fetching {len(queries_to_fetch)} uncached stations...")
            with tqdm(total=1, desc="Fetching satellite data") as pbar:
                results = retrieve_satellite_data(
                    queries=queries_to_fetch,
                    products=self.products,
                    spatial_pad=16,  # 33x33 region
                    temporal_pad=4,  # Include query day plus 4 previous days
                    max_batch_size=self.max_batch_size
                )
                pbar.update(1)
            
            # Process and cache the results
            for query_idx, station_idx in enumerate(station_indices):
                if query_idx in results:
                    station = stations[station_idx]
                    sat_data = {}
                    for product, data in results[query_idx].items():
                        for var, values in data['data'].items():
                            if var != 'time':
                                sat_data[f"{product}_{var}"] = values
                    
                    # Cache the results
                    cache_key = self._get_cache_key(
                        station['latitude'],
                        station['longitude'],
                        station['timestamp']
                    )
                    self.cache[cache_key] = sat_data
                    cached_results[station_idx] = sat_data
        
        # Save the updated cache
        self._save_cache()
        
        # Combine station data with satellite data
        stations_with_sat = []
        for i, station in enumerate(stations):
            station['satellite_data'] = cached_results.get(i)
            stations_with_sat.append(station)
        
        return stations_with_sat 