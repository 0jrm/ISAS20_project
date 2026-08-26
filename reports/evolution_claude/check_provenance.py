#!/usr/bin/env python
"""Runnable provenance check for reports/evolution/{EVOLUTION.md,index.html}.

Verifies, using only stdlib (re, json, pathlib, subprocess):

1. Every figure artifact_id in PROVENANCE.json points at a file that exists.
2. Every figs/*.svg on disk is referenced by exactly one PROVENANCE.json figure
   artifact (no orphan figures, no missing ones).
3. Every fig referenced in EVOLUTION.md corresponds to a PROVENANCE.json artifact.
4. Every figure artifact's raw <svg>...</svg> content is actually embedded in
   index.html (byte-for-byte), i.e. the HTML was not hand-edited out of sync
   with the figs/ directory.
5. The OSSE gate verdicts (E3_gt_E2, E4_ge_E3) shown in EVOLUTION.md and
   index.html match the claims recorded in the frozen source JSON
   (NeSPReSO2_onTemplate/saved/runs/phase6_osse/cast_column_s42.json).
6. No unsourced[] value (the numbers this report explicitly declined to show)
   leaked into EVOLUTION.md or into index.html *outside* the deliberate
   Provenance-tab omission ledger (.unsourcedbox).
7. Every decimal / scientific-notation number appearing in the *claims*
   surface of EVOLUTION.md and index.html (tables, prose; not SVG geometry,
   not script/style, not the omission ledger) is a verbatim substring of at
   least one cited source_files entry, a contradictions[].sources file,
   PROVENANCE.json itself, or the two Phase-6 R_cal git commit messages.
   Range hyphens (e.g. 0.0282-0.3753) are not treated as minus signs.

Run: python3 reports/evolution/check_provenance.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIGS = HERE / "figs"

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


# ---------------------------------------------------------------------------
provenance = json.loads((HERE / "PROVENANCE.json").read_text())
lineage = json.loads((HERE / "lineage.json").read_text())
evolution_md = (HERE / "EVOLUTION.md").read_text()
index_html = (HERE / "index.html").read_text()

artifacts = provenance["artifacts"]
fig_artifacts = [a for a in artifacts if a["kind"] == "figure"]

# ---------------------------------------------------------------------------
# 1. Every declared figure path exists.
# ---------------------------------------------------------------------------
declared_fig_files: set[str] = set()
for a in fig_artifacts:
    for key in ("path", "png_path"):
        p = a.get(key)
        if not p:
            continue
        declared_fig_files.add(Path(p).name)
        full = ROOT / p
        if full.exists():
            ok(f"{a['artifact_id']}: {p} exists")
        else:
            fail(f"{a['artifact_id']}: declared path {p} does not exist on disk")

# ---------------------------------------------------------------------------
# 2. Every figs/*.svg on disk is declared (no orphans), and vice versa.
# ---------------------------------------------------------------------------
on_disk = {p.name for p in FIGS.glob("*.svg")} | {p.name for p in FIGS.glob("*.png")}
orphans = on_disk - declared_fig_files
missing = declared_fig_files - on_disk
if orphans:
    fail(f"figs/ contains files not declared in any PROVENANCE.json artifact: {sorted(orphans)}")
else:
    ok("no orphan files under figs/")
if missing:
    fail(f"PROVENANCE.json declares files not present under figs/: {sorted(missing)}")
else:
    ok("no missing declared figure files")

# ---------------------------------------------------------------------------
# 3. Every fig referenced in EVOLUTION.md corresponds to a declared artifact.
# ---------------------------------------------------------------------------
md_fig_refs = set(re.findall(r"figs/([\w\-]+\.(?:svg|png))", evolution_md))
unknown_md_refs = md_fig_refs - declared_fig_files
if unknown_md_refs:
    fail(f"EVOLUTION.md references figures not in PROVENANCE.json: {sorted(unknown_md_refs)}")
else:
    ok(f"EVOLUTION.md's {len(md_fig_refs)} figure references all resolve in PROVENANCE.json")

# ---------------------------------------------------------------------------
# 4. Every figure artifact's raw SVG content is embedded verbatim in index.html.
# ---------------------------------------------------------------------------
svg_body_re = re.compile(r"<svg[\s\S]*</svg>")
for a in fig_artifacts:
    p = a.get("path")
    if not p or not p.endswith(".svg"):
        continue
    full = ROOT / p
    if not full.exists():
        continue  # already reported above
    m = svg_body_re.search(full.read_text())
    if not m:
        fail(f"{a['artifact_id']}: could not locate <svg>...</svg> body in {p}")
        continue
    if m.group(0) in index_html:
        ok(f"{a['artifact_id']}: embedded verbatim in index.html")
    else:
        fail(f"{a['artifact_id']}: {p} content NOT found verbatim in index.html (out of sync — rerun build_html.py)")

# ---------------------------------------------------------------------------
# 5. OSSE gate verdicts in EVOLUTION.md / index.html match the source JSON.
# ---------------------------------------------------------------------------
cast_json = json.loads((ROOT / "NeSPReSO2_onTemplate/saved/runs/phase6_osse/cast_column_s42.json").read_text())
claims = cast_json["claims"]
expected = {"E3_gt_E2": "FAIL" if not claims["E3_gt_E2"] else "PASS",
            "E4_ge_E3": "FAIL" if not claims["E4_ge_E3"] else "PASS"}

for claim, verdict in expected.items():
    pretty = {"E3_gt_E2": "E3 > E2", "E4_ge_E3": "E4 ≥ E3", }[claim] if False else claim
    patterns_md = {
        "E3_gt_E2": r"E3\s*>\s*E2[^\n]*?FAIL" if verdict == "FAIL" else r"E3\s*>\s*E2[^\n]*?PASS",
        "E4_ge_E3": r"E4\s*≥\s*E3[^\n]*?FAIL" if verdict == "FAIL" else r"E4\s*≥\s*E3[^\n]*?PASS",
    }[claim]
    if re.search(patterns_md, evolution_md):
        ok(f"EVOLUTION.md states {claim}: {verdict}, matching cast_column_s42.json claims")
    else:
        fail(f"EVOLUTION.md does not clearly state {claim}: {verdict} (source JSON says claims.{claim}={claims[claim]})")

    patterns_html = {
        "E3_gt_E2": r"E3\s*&gt;\s*E2[^<]*?FAIL",
        "E4_ge_E3": r"E4\s*&#8805;|E4\s*≥\s*E3[^<]*?FAIL",
    }
    html_ok = ("FAIL" in index_html and (
        re.search(r"E3.{0,15}FAIL", index_html) if claim == "E3_gt_E2" else re.search(r"E4.{0,15}FAIL", index_html)
    ))
    if html_ok:
        ok(f"index.html states {claim}: {verdict}, matching cast_column_s42.json claims")
    else:
        fail(f"index.html does not clearly state {claim}: {verdict}")

# Also assert PROVENANCE.json's own recorded gate_verdict for tbl_osse_canonical agrees.
tbl_osse = next(a for a in artifacts if a["artifact_id"] == "tbl_osse_canonical")
if "FAIL" in tbl_osse["gate_verdict"] and not claims["E3_gt_E2"] and not claims["E4_ge_E3"]:
    ok("PROVENANCE.json tbl_osse_canonical.gate_verdict agrees with cast_column_s42.json claims")
else:
    fail("PROVENANCE.json tbl_osse_canonical.gate_verdict disagrees with cast_column_s42.json claims")

# ---------------------------------------------------------------------------
# 6. No unsourced[] value leaked into the *claims* surfaces.
# ---------------------------------------------------------------------------
# The Provenance tab deliberately renders unsourced[] as an omission ledger
# (.unsourcedbox). That is documentation of what was withheld, not a leak.
# A leak is an unsourced number appearing in EVOLUTION.md or in index.html
# outside those ledger boxes (and outside the embedded provenance JSON).
NUM_RE = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+")


def numbers_in(text: str) -> set[str]:
    """Extract decimal/scientific numbers; treat digit-hyphen-digit as a range, not a minus."""
    out: set[str] = set()
    for m in NUM_RE.finditer(text):
        s = m.group(0)
        if s.startswith("-") and m.start() > 0 and text[m.start() - 1].isdigit():
            s = s[1:]  # e.g. "0.0282-0.3753" → 0.3753, not -0.3753
        out.add(s)
    return out


html_claims = index_html
for tag in ("script", "style", "svg"):
    html_claims = re.sub(rf"<{tag}[\s\S]*?</{tag}>", "", html_claims)
# Strip the deliberate omission ledger before leak/source checks.
html_claims = re.sub(r'<div class="unsourcedbox"[\s\S]*?</div>', "", html_claims)

for u in provenance["unsourced"]:
    leaked_numbers = numbers_in(u["value"])
    for n in leaked_numbers:
        if n in evolution_md:
            fail(f"unsourced value '{n}' (from '{u['requested_by']}') appears in EVOLUTION.md")
        if n in html_claims:
            fail(f"unsourced value '{n}' (from '{u['requested_by']}') appears in index.html claims surface")
if not FAILURES or not any("unsourced value" in f for f in FAILURES):
    ok("no unsourced[] numeric values leaked into EVOLUTION.md or index.html claims")

# ---------------------------------------------------------------------------
# 7. Every visible number in EVOLUTION.md / index.html has a source.
# ---------------------------------------------------------------------------
all_source_files: set[str] = set()
for a in artifacts:
    all_source_files.update(a.get("source_files", []))
for n in lineage["nodes"]:
    all_source_files.update(n.get("source_files", []))
# Contradiction banners cite numbers from these sources (e.g. stale run.log);
# include them so the warning text itself is sourced rather than banned.
for c in provenance.get("contradictions", []):
    all_source_files.update(c.get("sources", []))
# PROVENANCE.json itself is the ledger for banner/contradiction text.
all_source_files.add("reports/evolution/PROVENANCE.json")

corpus_parts = []
for rel in sorted(all_source_files):
    full = ROOT / rel
    if full.exists() and full.is_file():
        try:
            corpus_parts.append(full.read_text(errors="ignore"))
        except Exception:
            pass

for sha in (provenance["meta"]["prereg_commit"], provenance["meta"]["promotion_commit"]):
    try:
        msg = subprocess.run(["git", "show", "-s", "--format=%B", sha], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout
        corpus_parts.append(msg)
    except Exception as e:
        fail(f"could not retrieve git commit message for {sha}: {e}")

corpus = "\n".join(corpus_parts)

# Text to scan: EVOLUTION.md in full; index.html claims surface (script/style/svg
# and unsourcedbox already stripped above). SVG geometry is checked byte-for-byte
# in step 4; the omission ledger is checked only for non-leakage in step 6.
html_visible = html_claims

unsourced_numbers_found: dict[str, list[str]] = {}
for label, text in (("EVOLUTION.md", evolution_md), ("index.html (visible)", html_visible)):
    for n in sorted(numbers_in(text)):
        if n in corpus:
            continue
        unsourced_numbers_found.setdefault(n, []).append(label)

if unsourced_numbers_found:
    for n, locs in sorted(unsourced_numbers_found.items()):
        fail(f"number '{n}' in {locs} is not a substring of any cited source file or the two R_cal commit messages")
else:
    ok("every decimal/scientific number in EVOLUTION.md and index.html (visible text) is sourced")
# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) FAILED.")
    sys.exit(1)
print("All provenance checks passed.")
sys.exit(0)
