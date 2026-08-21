"""Hard inventory: machine paths in Python stay frozen. New ones fail selfcheck.

Why: AGENTS.md already bans hardcoded FS paths; agents keep pasting /unity anyway.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
_NEEDLES = ("/unity/", "/home/")

# Grandfathered files. Count is AST string literals containing a needle.
# Drop a file from this dict when you delete its last hardcoded path.
# Do not raise a count. Put new paths in config JSON.
MACHINE_PATH_COUNTS: dict[str, int] = {
    "diagnostics/stale_sat/cmp_sat_sources.py": 2,
    "diagnostics/stale_sat/diag_patch.py": 2,
    "diagnostics/stale_sat/e0_point_equiv.py": 2,
    "diagnostics/stale_sat/h5_stale_check.py": 1,
    "diagnostics/stale_sat/make_std_cache.py": 2,
    "diagnostics/stale_sat/split_vs_stale.py": 2,
    "notebooks/build_notebook.py": 1,
    "preproc/cube/cube_schema.py": 4,
    "preproc/preproc_argo.py": 4,
    "preproc/preproc_isas_sat.py": 1,
    "scripts/backend_equivalence.py": 1,
    "scripts/diagnose_e_deep_band.py": 1,
    "scripts/m2_spotcheck.py": 1,
    "scripts/t1_basis_stability.py": 1,
    "selfcheck.py": 3,
}

# Package layers are importable without sys.path hacks. Keep them that way.
NO_SYSPATH_PREFIXES = ("model/", "trainer/", "data_loader/", "evalphys/", "dacov/")


_SKIP_DIRS = {"saved", "data", "__pycache__", ".git"}


def _iter_py(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py") and name != "garden_gate.py":
                yield Path(dirpath) / name


def machine_path_counts(root: Path = PKG_ROOT) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in _iter_py(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        n = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(s in node.value for s in _NEEDLES):
                    n += 1
        if n:
            out[str(path.relative_to(root))] = n
    return out


def check_machine_paths(root: Path = PKG_ROOT) -> list[str]:
    found = machine_path_counts(root)
    errors: list[str] = []
    for rel, n in sorted(found.items()):
        allowed = MACHINE_PATH_COUNTS.get(rel)
        if allowed is None:
            errors.append(f"new hardcoded machine path in {rel} ({n} literals). Put it in config JSON.")
        elif n > allowed:
            errors.append(f"{rel}: machine-path literals {n} > frozen {allowed}")
    return errors


def check_core_syspath(root: Path = PKG_ROOT) -> list[str]:
    errors: list[str] = []
    for path in _iter_py(root):
        rel = str(path.relative_to(root))
        if not rel.startswith(NO_SYSPATH_PREFIXES):
            continue
        text = path.read_text(encoding="utf-8")
        if "sys.path.insert" in text or "sys.path.append" in text:
            errors.append(f"sys.path mutation in package layer {rel}")
    return errors


def run_garden_gate(root: Path = PKG_ROOT) -> None:
    errors = check_machine_paths(root) + check_core_syspath(root)
    if errors:
        raise AssertionError("garden_gate:\n" + "\n".join(errors))


if __name__ == "__main__":
    run_garden_gate()
    print("garden_gate: ok")
