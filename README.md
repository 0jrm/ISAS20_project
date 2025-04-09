# ISAS20 Project

This repository contains utilities for working with ISAS20 and ARGO data.

## Module Structure

The repository is organized as a Python package with the following structure:

```
ISAS20_project/
├── utils/
│   ├── __init__.py
│   └── retrieve_sat.py
└── README.md
```

## Using the Utility Modules

### Satellite Data Retrieval

The `retrieve_sat.py` module provides functions to retrieve and interpolate satellite data for oceanographic applications. 

Example usage:

```python
from ISAS20_project.utils import retrieve_satellite_data

# Define queries (latitude, longitude, julian_date)
queries = [
    (45.0, -30.0, 2459020.5),
    (45.5, -29.5, 2459020.5)
]

# Define the satellite products and variables to extract
products = {
    "bathymetry": ["elevation"],
    "ostia": ["analysed_sst"],
    "sss": ["sos"]
}

# Set parameters
spatial_padding = 16  # extract a 33x33 region
temporal_padding = 0  # No temporal padding

# Retrieve data
results = retrieve_satellite_data(queries, products, spatial_padding, temporal_padding)

# Access results
for idx, res in results.items():
    print(f"\nQuery {idx}: {queries[idx]}")
    for prod, info in res.items():
        print(f"  Product: {prod}")
        for var, data in info["data"].items():
            if var != "time":
                print(f"    Variable '{var}': data shape = {data.shape}")
```

## Available Products

The module supports the following satellite products:

- `bathymetry`: Ocean floor elevation data
- `ostia`: OSTIA sea surface temperature
- `remss`: REMSS sea surface temperature
- `wind`: Wind data (windspeed, u_wind, v_wind)
- `ssh`: Sea surface height (adt, sla, ugos, vgos)
- `sss`: Sea surface salinity (sos) 