#!/usr/bin/env python3
"""Phase 2.2 smoke: fetch verified error fields for 3 GoM stations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from retrieve_sat import retrieve_satellite_data

_HERE = Path(__file__).resolve().parent
_V3 = json.loads((_HERE / "v3.json").read_text())

# Three GoM ARGO-ish points (lat, lon, JULD ~ mid-2018)
QUERIES = [
    (25.5, -90.0, 2458285.0),
    (27.0, -92.5, 2458300.0),
    (24.0, -85.5, 2458315.0),
]

PRODUCTS = {
    "ssh": ["err_sla"],
    "ostia": ["analysis_error"],
    "sss": ["sos_error"],
}


def main() -> int:
    assert _V3["error_channels"] == {
        "ostia": "analysis_error",
        "sss": "sos_error",
        "ssh": "err_sla",
    }
    results = retrieve_satellite_data(
        QUERIES,
        PRODUCTS,
        spatial_pad=2,
        temporal_pad=2,
    )
    assert len(results) == 3
    for i in sorted(results.keys()):
        row = results[i]
        for prod, var in (
            ("ssh", "err_sla"),
            ("ostia", "analysis_error"),
            ("sss", "sos_error"),
        ):
            assert prod in row, f"station {i}: missing product {prod}"
            data = row[prod]["data"]
            assert var in data, f"station {i}: missing {prod}/{var}; keys={list(data)}"
            arr = np.asarray(data[var], dtype=np.float64)
            finite = np.isfinite(arr)
            print(
                f"station {i} {prod}/{var}: shape={arr.shape} "
                f"finite={finite.mean():.2%} "
                f"median={np.nanmedian(arr):.4g}"
            )
            assert finite.any(), f"station {i}: all-NaN {prod}/{var}"
    print("check_v3_errors: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
