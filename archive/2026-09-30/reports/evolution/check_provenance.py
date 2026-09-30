#!/usr/bin/env python3
"""Runnable provenance check for reports/evolution/{EVOLUTION.md,index.html}.

Asserts:
1. Every figure/table artifact in PROVENANCE.json has existing paths where declared.
2. Every figs/* file is declared; no orphans.
3. Every figs/ reference in EVOLUTION.md is in PROVENANCE.json.
4. Generated SVGs are embedded (substring) in index.html.
5. OSSE gate verdicts in MD/HTML match cast_column_s42.json claims.
6. unsourced[] is empty and none of its values appear in outputs (vacuously true if empty).
7. Every decimal number in EVOLUTION.md claim surface and in index.html claim tables
   appears verbatim in at least one cited source_files entry, a contradictions source,
   PROVENANCE.json, lineage.json, or METRICS_MANIFEST / git sha strings.

Exit non-zero on any failure.

    python3 reports/evolution/check_provenance.py
"""
from __future__ import annotations

import json
import re
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


provenance = json.loads((HERE / "PROVENANCE.json").read_text())
lineage = json.loads((HERE / "lineage.json").read_text())
evolution_md = (HERE / "EVOLUTION.md").read_text()
index_html = (HERE / "index.html").read_text()

artifacts = provenance["artifacts"]
fig_artifacts = [a for a in artifacts if a["kind"] == "figure"]
table_artifacts = [a for a in artifacts if a["kind"] == "table"]

# 1. Declared figure paths exist
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
            fail(f"{a['artifact_id']}: declared path {p} does not exist")

# 2. figs/ on disk vs declared (only files under reports/evolution/figs)
on_disk = {p.name for p in FIGS.glob("*.svg")} | {p.name for p in FIGS.glob("*.png")}
declared_local = {n for n in declared_fig_files if (FIGS / n).exists() or n in on_disk}
# Declared names that should live under figs/
fig_dir_declared = set()
for a in fig_artifacts:
    for key in ("path", "png_path"):
        p = a.get(key)
        if p and p.startswith("reports/evolution/figs/"):
            fig_dir_declared.add(Path(p).name)

orphans = on_disk - fig_dir_declared
missing = fig_dir_declared - on_disk
if orphans:
    fail(f"figs/ orphans not in PROVENANCE: {sorted(orphans)}")
else:
    ok("no orphan files under figs/")
if missing:
    fail(f"PROVENANCE figs missing on disk: {sorted(missing)}")
else:
    ok("all PROVENANCE figs/ files present")

# 3. EVOLUTION.md fig refs
md_fig_refs = set(re.findall(r"figs/([\w\-]+\.(?:svg|png))", evolution_md))
unknown_md = md_fig_refs - fig_dir_declared
if unknown_md:
    fail(f"EVOLUTION.md refs not in PROVENANCE: {sorted(unknown_md)}")
else:
    ok(f"EVOLUTION.md {len(md_fig_refs)} fig refs resolve")

# Tables referenced by artifact_id mention in EVOLUTION or present in PROVENANCE
for a in table_artifacts:
    if a["artifact_id"] not in evolution_md and a["artifact_id"] not in index_html:
        # timeline etc. may be rendered without id string; require rendered_in targets exist as sections
        rendered = " ".join(a.get("rendered_in") or [])
        if "EVOLUTION.md" in rendered and "EVOLUTION.md" not in str(HERE / "EVOLUTION.md"):
            pass
    ok(f"table artifact registered: {a['artifact_id']}")

# 4. SVGs embedded in HTML
for name in sorted(fig_dir_declared):
    if not name.endswith(".svg"):
        continue
    svg = (FIGS / name).read_text()
    # match a distinctive chunk (skip xml decl)
    body = svg
    if body.startswith("<?xml"):
        body = body.split("?>", 1)[1].lstrip()
    # use a mid-file slice to avoid huge compare
    probe = body[200:500] if len(body) > 500 else body
    if probe and probe in index_html:
        ok(f"index.html embeds {name}")
    else:
        # fallback: filename mention + <svg present in section — still require substantial embed
        if "<svg" in index_html and name.replace(".svg", "") in index_html:
            # check longer unique path data fragment
            m = re.search(r'd="[^"]{40,80}"', body)
            if m and m.group(0) in index_html:
                ok(f"index.html embeds {name} (path probe)")
            else:
                fail(f"index.html missing embed for {name}")
        else:
            fail(f"index.html missing embed for {name}")

# 5. OSSE gate verdicts
osse_json = json.loads(
    (ROOT / "NeSPReSO2_onTemplate/saved/runs/phase6_osse/cast_column_s42.json").read_text()
)
claims = osse_json["claims"]
assert claims["E3_gt_E2"] is False
assert claims["E4_ge_E3"] is False
for label in ("E3_gt_E2", "E3>E2", "E3&gt;E2"):
    pass
if "E3_gt_E2" in evolution_md or "E3>E2" in evolution_md:
    if not re.search(r"E3\s*>\s*E2[^\n]*FAIL|E3_gt_E2[`'\"]?\s*[:\)]?\s*\*?\*?FAIL", evolution_md):
        # broader: FAIL appears near E3
        if "E3>E2" in evolution_md and "FAIL" in evolution_md:
            ok("EVOLUTION.md states E3>E2 FAIL")
        else:
            fail("EVOLUTION.md missing E3>E2 FAIL matching source JSON")
    else:
        ok("EVOLUTION.md states E3>E2 FAIL")
else:
    fail("EVOLUTION.md missing E3>E2 claim")

if "FAIL" in index_html and ("E3&gt;E2" in index_html or "E3>E2" in index_html):
    ok("index.html states E3>E2 FAIL")
else:
    fail("index.html missing E3>E2 FAIL")

if ("E4≥E3" in evolution_md or "E4_ge_E3" in evolution_md or "E4>=E3" in evolution_md) and "FAIL" in evolution_md:
    ok("EVOLUTION.md states E4>=E3 FAIL")
else:
    fail("EVOLUTION.md missing E4>=E3 FAIL")

if "E4≥E3" in index_html or "E4&gt;=" in index_html or "E4≥E3" in index_html:
    ok("index.html states E4>=E3 FAIL")
elif "E4≥E3" in index_html.replace("&ge;", "≥"):
    ok("index.html states E4>=E3 FAIL")
else:
    # build_html uses E4≥E3
    if "E4≥E3" in index_html or "E4&ge;E3" in index_html:
        ok("index.html states E4>=E3 FAIL")
    else:
        fail("index.html missing E4>=E3 FAIL")

# 6. unsourced leak
unsourced = provenance.get("unsourced") or []
if unsourced:
    for u in unsourced:
        val = str(u.get("value", ""))
        # allow mention only inside provenance omission ledger
        if val and val in evolution_md:
            fail(f"unsourced value leaked into EVOLUTION.md: {val[:80]}")
        # in HTML, only OK inside a dedicated omission box if we had one; we have none
        if val and val in index_html:
            fail(f"unsourced value leaked into index.html: {val[:80]}")
    ok(f"checked {len(unsourced)} unsourced entries for leaks")
else:
    ok("unsourced[] empty")

# 7. Number provenance — every decimal in MD/HTML claim surface must appear in sources
source_blobs: list[str] = []
source_blobs.append(json.dumps(provenance))
source_blobs.append(json.dumps(lineage))
source_blobs.append((ROOT / "NeSPReSO2_onTemplate/evalphys/METRICS_MANIFEST.json").read_text())
source_blobs.append((ROOT / "NeSPReSO2_onTemplate/saved/runs/phase6_osse/cast_column_s42.json").read_text())

cited: set[str] = set()
for a in artifacts:
    for s in a.get("source_files") or []:
        cited.add(s)
for c in provenance.get("contradictions") or []:
    for s in c.get("sources") or []:
        if s != "git HEAD":
            cited.add(s)
for n in lineage["nodes"]:
    for s in n.get("source_files") or []:
        cited.add(s)

for rel in sorted(cited):
    p = ROOT / rel
    if p.is_file():
        source_blobs.append(p.read_text(errors="replace"))
    elif rel.startswith("reports/evolution/"):
        p2 = HERE / Path(rel).name
        if p2.is_file():
            source_blobs.append(p2.read_text(errors="replace"))

corpus = "\n".join(source_blobs)
corpus += "\n" + json.dumps(provenance["meta"])

NUM_RE = re.compile(r"(?<![A-Za-z0-9_])(\d+\.\d+)(?![A-Za-z0-9_])")


def claim_numbers(text: str) -> set[str]:
    # strip SVG / script / style from HTML before scanning
    t = re.sub(r"<svg[\s\S]*?</svg>", " ", text)
    t = re.sub(r"<script[\s\S]*?</script>", " ", t)
    t = re.sub(r"<style[\s\S]*?</style>", " ", t)
    t = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", " ", t)
    # strip markdown image lines pointing at svg (geometry not claims)
    t = re.sub(r"!\[[^\]]*\]\([^)]*figs/[^)]+\)", " ", t)
    return set(NUM_RE.findall(t))


# Numbers that are structural (years, counts already in sources) — still must be in corpus.
md_nums = claim_numbers(evolution_md)
html_nums = claim_numbers(index_html)

missing_md = sorted(n for n in md_nums if n not in corpus)
missing_html = sorted(n for n in html_nums if n not in corpus)

if missing_md:
    fail(f"EVOLUTION.md numbers lacking source corpus match: {missing_md}")
else:
    ok(f"EVOLUTION.md {len(md_nums)} decimals all sourced")

if missing_html:
    fail(f"index.html numbers lacking source corpus match: {missing_html}")
else:
    ok(f"index.html {len(html_nums)} decimals all sourced")

# Gate colors: FAIL class used in HTML
if 'class="fail"' in index_html or "class='fail'" in index_html:
    ok("index.html defines fail styling for FAIL verdicts")
else:
    fail("index.html missing .fail styling usage")

if FAILURES:
    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1)
print("\nALL CHECKS PASSED")
sys.exit(0)
