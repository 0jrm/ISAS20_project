#!/usr/bin/env python3
"""Regression: refuse resume into old-schema batch dirs when error vars are required."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np

from generate_argo_satellite_data import (
    BATCH_PREFIX,
    assert_batch_dir_matches_products,
    batch_path,
)


def _write_old_schema_batch(path: Path) -> None:
    """Synthetic v2-era batch: value channels only, no error datasets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["n_stations"] = 2
        meta.attrs["batch_start"] = 0
        meta.attrs["batch_end"] = 2
        for prod, var in (("sss", "sos"), ("ostia", "analysed_sst"), ("ssh", "adt")):
            g = f.create_group(prod)
            g.create_dataset("latitude_grid", data=np.zeros(5))
            g.create_dataset("longitude_grid", data=np.zeros(5))
            g.create_dataset(var, data=np.zeros((2, 7, 5, 5), dtype=np.float32))
        st = f.create_group("stations")
        st.create_dataset("latitude", data=[25.0, 26.0])
        st.create_dataset("longitude", data=[-90.0, -91.0])
        st.create_dataset("julian_date", data=[2458000.0, 2458001.0])
        st.create_dataset("source_file", data=np.array([b"x", b"y"], dtype="S"))
        st.create_dataset("profile_index", data=[0, 1])


def test_resume_refuses_old_schema_when_errors_required() -> None:
    products_v3 = {
        "bathymetry": ["elevation"],
        "sss": ["sos", "sos_error"],
        "ostia": ["analysed_sst", "analysis_error"],
        "ssh": ["adt", "err_sla"],
    }
    with tempfile.TemporaryDirectory() as td:
        batch_dir = Path(td)
        bsize = 16
        # name must match list_batch_files pattern
        old = Path(batch_path(str(batch_dir), 0, 16, bsize))
        _write_old_schema_batch(old)
        try:
            assert_batch_dir_matches_products(str(batch_dir), bsize, products_v3)
        except RuntimeError as exc:
            msg = str(exc)
            assert "Regenerate from scratch or use v2 config" in msg, msg
            assert "sos_error" in msg or "analysis_error" in msg or "err_sla" in msg
        else:
            raise AssertionError("expected RuntimeError on schema mismatch")


def test_resume_ok_when_schema_matches() -> None:
    products_v2 = {
        "sss": ["sos"],
        "ostia": ["analysed_sst"],
        "ssh": ["adt"],
    }
    with tempfile.TemporaryDirectory() as td:
        batch_dir = Path(td)
        bsize = 16
        old = Path(batch_path(str(batch_dir), 0, 16, bsize))
        _write_old_schema_batch(old)
        assert_batch_dir_matches_products(str(batch_dir), bsize, products_v2)


if __name__ == "__main__":
    test_resume_refuses_old_schema_when_errors_required()
    test_resume_ok_when_schema_matches()
    print("test_batch_schema_guard: OK")
