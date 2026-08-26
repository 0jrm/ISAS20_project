#!/usr/bin/env python
"""Assemble reports/evolution/index.html: one self-contained file.

Embeds lineage.json and PROVENANCE.json inline, inlines every SVG in figs/,
and renders the summary tables. No external network calls, no build step,
no localStorage. Run with any Python 3 (stdlib only):

    python3 reports/evolution/build_html.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"

PROVENANCE = json.loads((HERE / "PROVENANCE.json").read_text())
LINEAGE = json.loads((HERE / "lineage.json").read_text())


def inline_svg(name: str) -> str:
    text = (FIGS / f"{name}.svg").read_text()
    # Strip the XML/DOCTYPE preamble matplotlib emits; keep from <svg ...> onward.
    m = re.search(r"<svg[\s\S]*</svg>", text)
    svg = m.group(0) if m else text
    # Give each embedded svg a scoped id-free namespace is unnecessary here
    # (single-document embed); just wrap for CSS sizing.
    return f'<div class="fig">{svg}</div>'


# ---------------------------------------------------------------------------
# Data tables (verbatim from the same sourced numbers as PROVENANCE.json)
# ---------------------------------------------------------------------------

MATRIX_LATENT_ROWS = [
    ("A: separate T/S PCA-16", "det", "T=0.541 ± 0.004", "PASS (≤0.5903)", False),
    ("A: separate T/S PCA-16", "CRPS", "CRPS=1.237 ± 0.010, ENCE=0.053 ± 0.004", "PASS (ENCE<0.20)", True),
    ("A: separate T/S PCA-16", "NLL", "CRPS=1.257 ± 0.031, ENCE=0.052 ± 0.004", "PASS (ENCE<0.20)", True),
    ("B: joint T/S EOF-32", "det", "T=0.534 ± 0.001", "PASS (≤0.5903)", False),
    ("B: joint T/S EOF-32", "CRPS", "CRPS=2.761 ± 0.028, ENCE=0.069 ± 0.004", "PASS (ENCE<0.20)", True),
    ("B: joint T/S EOF-32", "NLL", "CRPS=2.754 ± 0.008, ENCE=0.082 ± 0.012", "PASS (ENCE<0.20)", True),
    ("C: monotone-ρ + spice PCA", "det", "T=0.609 ± 0.012", "FAIL (>0.5903)", False),
    ("C: monotone-ρ + spice PCA", "CRPS", "CRPS=0.742 ± 0.048, ENCE=0.248 ± 0.021", "FAIL (ENCE≥0.20)", True),
    ("C: monotone-ρ + spice PCA", "NLL", "CRPS=0.774 ± 0.036, ENCE=0.120 ± 0.021", "PASS (ENCE<0.20)", True),
]

MATRIX_PHYSICAL_ROWS = [
    ("A×CRPS", "0.559±0.005", "0.119±0.001", "0.153±0.007", "dissertation default"),
    ("A×NLL", "0.575±0.023", "0.122±0.005", "0.162±0.019", "ENCE survivor, not picked"),
    ("A×det", "0.541±0.004", "—", "—", "det survivor"),
    ("B×CRPS", "0.586±0.054", "0.133±0.009", "0.247±0.003", "ENCE FAIL"),
    ("B×NLL", "0.563±0.011", "0.128±0.002", "0.299±0.013", "ENCE FAIL"),
    ("B×det", "0.534±0.001", "—", "—", "best det-only skill in matrix"),
    ("C×CRPS", "0.618±0.103", "0.139±0.022", "0.384±0.010", "ENCE FAIL"),
    ("C×NLL", "0.694±0.081", "0.157±0.018", "0.397±0.011", "ENCE FAIL"),
    ("C×det", "0.609±0.012", "—", "—", "skill-floor FAIL"),
]

OSSE_ROWS = [
    ("E0", "none", "1.5382", "—"),
    ("E1", "R_fixed_clim", "1.5382", "—"),
    ("E2", "R_fixed_isop", "0.5410", "—"),
    ("E3", "R_fixed_nespreso", "0.5454", "—"),
    ("E4", "R_cal (full localized)", "0.6160", "—"),
    ("E5", "R_cal + QC (keep 0.444 fraction)", "1.4008", "0.444"),
]

RCAL_DIAG_ROWS = [
    ("E3", "R_fixed (no calibration)", "0.5454"),
    ("E4 --rcal diag", "v1 fallback, diag(Σ_T) only", "0.5463"),
    ("E4 --rcal full", "v2, full localized Σ_T", "0.6160"),
]

TIMELINE_ROWS = [
    ("Phase 0", "What metric suite can every later phase trust?", "Freeze evalphys v1.1.0.", "survivor", "NeSPReSO2_onTemplate/evalphys/METRICS_MANIFEST.json"),
    ("Phase 1 (T2)", "Is the chronological test split stale-satellite contaminated?", "Stale-fraction audit by split/channel.", "survivor (gate OPEN)", "reports/stale_by_split.md"),
    ("Phase 1 (T1)", "Do soft basis changes (B, C) cut violations ≥5× vs A?", "Reconstruct truth through A/B/C/D; measure σ₀/N² violations.", "killed (escalated)", "reports/t1_basis_stability.md"),
    ("Phase 1 (R1)", "Does a hard monotone constraint (D) succeed where soft bases fail?", "softplus+cumsum control-grid density head.", "survivor (human sign-off)", "reports/t1_basis_stability.md"),
    ("Phase 3", "Does the hard monotone head recover skill at acceptable cost?", "σ₀ control-grid head + spice PCA-16 + Newton inversion.", "survivor", "PLAN-v2-recovery.md"),
    ("Phase 3 (erratum)", "Was the chronological gate evaluated cleanly?", "Found argo16_scales is random-split; chrono evals of it were leaked-optimistic.", "erratum", "reports/gate_floor_provenance.md"),
    ("Phase 3", "Does σ₀-space low-rank compression preserve skill (vs a-space)?", "PCA-16 on (σ₀-clim) + isotonic-at-inference.", "survivor (σ₀-space); killed (a-space)", "reports/finding_compress_physical_space.md"),
    ("Phase 4", "Does a two-stage CRPS/NLL head produce calibrated uncertainty?", "Heteroscedastic head; s2b + val per-dim recalibration.", "survivor (s2b)", "reports/phase4_ence_recalib_s2b.md"),
    ("Phase 5", "Under a fair 3-seed protocol, which cell wins?", "3×3 matrix; protocol v1 killed, v2 scored all 9 cells.", "survivor (v2 matrix)", "reports/ablation_summary.md"),
    ("Phase 5", "Does the C×det admission pass (0.562) replicate?", "Scored under locked protocol v2, 3 seeds.", "killed (0.609±0.012)", "reports/finding_C_det_gate_overfit.md"),
    ("Phase 5", "Does the latent ranking survive a physical decode?", "Rescore all 9 cells after decode to (T,S).", "superseded", "reports/ablation_summary.md"),
    ("Phase 5", "Which cell is the dissertation default?", "Apply §3 decision rule to the physical table.", "survivor: A×CRPS", "reports/ablation_summary.md"),
    ("Phase 5", "Does the winner calibrate uniformly by depth/season?", "Strata scoring of A×CRPS in physical T space.", "killed (ENCE(T)=0.236)", "reports/phase5_A_CRPS_physical_strata.md"),
    ("Phase 6", "Pre-register OSSE before winner identity is known.", "E0–E5 table, R construction, QC rule locked.", "survivor", "reports/osse_preregistration.md"),
    ("Phase 6", "Do NeSPReSO casts beat ISOP, and does calibrated R help?", "cast-column v1 OSSE, R_cal=diag(Σ_T).", "killed (both FAIL)", "reports/osse_results.md"),
    ("Phase 6", "Does full localized Σ_T restore stability with useful structure?", "Pre-register full localized Σ_T, L_loc=150m.", "survivor (prereg)", "reports/osse_preregistration.md"),
    ("Phase 6", "Does the structured covariance beat the diagonal fallback?", "Canonical run with pre-registered full R_cal.", "killed (0.6160 vs 0.5463)", "reports/osse_results.md"),
]


def esc(s):
    return html.escape(str(s))


def verdict_class(v: str) -> str:
    v = v.upper()
    if "FAIL" in v or v == "KILLED":
        return "v-fail"
    if "PASS" in v or v == "SURVIVOR":
        return "v-pass"
    if v == "SUPERSEDED":
        return "v-superseded"
    if v == "ERRATUM":
        return "v-erratum"
    return ""


def table(headers, rows, klass=""):
    out = [f'<table class="datatable {klass}"><thead><tr>']
    for h in headers:
        out.append(f"<th>{esc(h)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for i, cell in enumerate(row):
            cls = ""
            if isinstance(cell, str) and (("PASS" in cell and "FAIL" not in cell) or "FAIL" in cell):
                cls = ' class="' + verdict_class(cell) + '"'
            out.append(f"<td{cls}>{esc(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def timeline_table():
    rows = []
    for phase, q, change, verdict, src in TIMELINE_ROWS:
        rows.append((phase, q, change, verdict, src))
    out = ['<table class="datatable"><thead><tr><th>phase</th><th>question</th><th>change</th>'
           '<th>verdict</th><th>source</th></tr></thead><tbody>']
    for phase, q, change, verdict, src in rows:
        cls = verdict_class(verdict)
        out.append(f"<tr><td>{esc(phase)}</td><td>{esc(q)}</td><td>{esc(change)}</td>"
                    f'<td class="{cls}">{esc(verdict)}</td><td><code>{esc(src)}</code></td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def matrix_latent_table():
    out = ['<table class="datatable"><thead><tr><th>representation</th><th>head</th>'
           '<th>judgment number</th><th>gate</th><th>comparable across reps?</th></tr></thead><tbody>']
    for rep, head, text, gate, hatched in MATRIX_LATENT_ROWS:
        cls = verdict_class(gate)
        comp = "NO — PC-space CRPS/NLL" if hatched else "yes (physical T RMSE)"
        out.append(f"<tr><td>{esc(rep)}</td><td>{esc(head)}</td><td>{esc(text)}</td>"
                    f'<td class="{cls}">{esc(gate)}</td><td>{esc(comp)}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def matrix_physical_table():
    return table(["cell", "T RMSE", "physical CRPS(T+S)", "physical ENCE", "note"], MATRIX_PHYSICAL_ROWS)


def osse_table():
    return table(["E", "R construction", "overall T RMSE", "retention"], OSSE_ROWS)


def rcal_diag_table():
    return table(["variant", "R construction", "overall T RMSE"], RCAL_DIAG_ROWS)


def provenance_table():
    out = ['<table class="datatable prov"><thead><tr><th>artifact_id</th><th>kind</th>'
           '<th>source_files</th><th>git_sha</th><th>judgment_number</th><th>gate_verdict</th>'
           '<th>caveats</th></tr></thead><tbody>']
    for a in PROVENANCE["artifacts"]:
        srcs = "<br>".join(f"<code>{esc(s)}</code>" for s in a.get("source_files", []))
        caveats = "<ul>" + "".join(f"<li>{esc(c)}</li>" for c in a.get("caveats", [])) + "</ul>" if a.get("caveats") else ""
        gv = a.get("gate_verdict") or ""
        cls = verdict_class(gv) if gv else ""
        sha = a.get("git_sha") or "—"
        jn = a.get("judgment_number")
        jn = "—" if jn is None else jn
        out.append("<tr>")
        out.append(f'<td><code>{esc(a["artifact_id"])}</code></td>')
        out.append(f'<td>{esc(a["kind"])}</td>')
        out.append(f"<td>{srcs}</td>")
        out.append(f"<td><code>{esc(sha)}</code></td>")
        out.append(f"<td>{esc(jn)}</td>")
        out.append(f'<td class="{cls}">{esc(gv)}</td>')
        out.append(f"<td>{caveats}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def contradictions_block():
    out = []
    for c in PROVENANCE["contradictions"]:
        out.append(f'<div class="warnbox"><strong>{esc(c["id"])}</strong><p>{esc(c["description"])}</p>'
                    f'<p><em>Sources:</em> ' + ", ".join(f"<code>{esc(s)}</code>" for s in c["sources"]) + "</p>"
                    f'<p><em>Handling:</em> {esc(c["handling"])}</p></div>')
    return "".join(out)


def unsourced_block():
    out = []
    for u in PROVENANCE["unsourced"]:
        out.append(f'<div class="unsourcedbox"><strong>Requested by:</strong> {esc(u["requested_by"])}<br>'
                    f'<strong>Value not shown:</strong> {esc(u["value"])}<br>'
                    f'<strong>Reason:</strong> {esc(u["reason"])}<br>'
                    f'<strong>Action taken:</strong> {esc(u["action"])}</div>')
    return "".join(out)


CSS = """
:root { --pass:#2a7f3f; --fail:#b3261e; --superseded:#b8860b; --erratum:#5a4fcf; --bg:#ffffff; --fg:#1a1a1a; --panel:#f7f7f8; --border:#dddddd; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#15171a; --fg:#e8e8e8; --panel:#1f2226; --border:#3a3d42; }
}
:root[data-theme="dark"] { --bg:#15171a; --fg:#e8e8e8; --panel:#1f2226; --border:#3a3d42; }
:root[data-theme="light"] { --bg:#ffffff; --fg:#1a1a1a; --panel:#f7f7f8; --border:#dddddd; }
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; background:var(--bg); color:var(--fg); }
header.top { position:sticky; top:0; z-index:10; background:var(--panel); border-bottom:1px solid var(--border); padding:10px 18px; font-size:12.5px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
header.top b { color:var(--fail); }
nav.tabs { display:flex; gap:4px; padding:10px 18px 0; background:var(--panel); border-bottom:1px solid var(--border); flex-wrap:wrap; }
nav.tabs button { background:none; border:1px solid var(--border); border-bottom:none; padding:8px 16px; cursor:pointer; color:var(--fg); border-radius:6px 6px 0 0; font-size:14px; }
nav.tabs button.active { background:var(--bg); font-weight:600; }
main { max-width:1200px; margin:0 auto; padding:22px 18px 60px; }
section.tab { display:none; }
section.tab.active { display:block; }
h1,h2,h3 { line-height:1.25; }
p { line-height:1.5; max-width:900px; }
.warnbanner { background:#fff3cd; border:1px solid #e0b400; color:#5c4600; padding:10px 14px; border-radius:6px; margin-bottom:14px; font-size:14px; }
:root[data-theme="dark"] .warnbanner, @media (prefers-color-scheme: dark) { }
.warnbox { border:1px solid var(--fail); background:rgba(179,38,30,0.08); border-radius:6px; padding:10px 14px; margin:10px 0; font-size:14px; }
.unsourcedbox { border:1px dashed var(--superseded); background:rgba(184,134,11,0.08); border-radius:6px; padding:10px 14px; margin:10px 0; font-size:14px; }
table.datatable { border-collapse: collapse; width:100%; margin:14px 0; font-size:13px; }
table.datatable th, table.datatable td { border:1px solid var(--border); padding:6px 8px; text-align:left; vertical-align:top; }
table.datatable th { background:var(--panel); }
table.prov td { max-width:260px; overflow-wrap:anywhere; }
.v-pass { color:var(--pass); font-weight:700; }
.v-fail { color:var(--fail); font-weight:700; }
.v-superseded { color:var(--superseded); font-weight:700; }
.v-erratum { color:var(--erratum); font-weight:700; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; background:rgba(127,127,127,0.12); padding:1px 4px; border-radius:3px; }
.fig { overflow-x:auto; margin:16px 0; border:1px solid var(--border); border-radius:6px; padding:8px; background:var(--panel); }
.fig svg { max-width:100%; height:auto; display:block; margin:0 auto; }
.figgrid { display:grid; grid-template-columns:1fr; gap:10px; }
#dagwrap { display:flex; gap:16px; flex-wrap:wrap; }
#daghost { flex: 2 1 640px; overflow:auto; border:1px solid var(--border); border-radius:6px; background:var(--panel); }
#dagpanel { flex: 1 1 320px; border:1px solid var(--border); border-radius:6px; padding:14px; background:var(--panel); min-height:220px; }
#dagpanel h3 { margin-top:0; }
.node-rect { cursor:pointer; }
.node-rect:hover { stroke-width:3.5; }
.legend-item { display:inline-flex; align-items:center; gap:5px; margin-right:14px; font-size:12px; }
.legend-swatch { width:14px; height:14px; border-radius:3px; display:inline-block; border:2px solid; }
footer { padding:20px 18px; text-align:center; font-size:12px; color:#888; }
"""

JS = r"""
function showTab(id) {
  document.querySelectorAll('section.tab').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('nav.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  document.getElementById('btn-' + id).classList.add('active');
  if (id === 'lineage') renderDag();
}

const VERDICT_COLOR = { survivor: '#2a7f3f', killed: '#b3261e', superseded: '#b8860b', erratum: '#5a4fcf' };
const EDGE_STYLE = {
  led_to: { stroke: '#888888', dash: 'none' },
  superseded_by: { stroke: '#b8860b', dash: '6,4' },
  erratum_of: { stroke: '#5a4fcf', dash: '2,3' },
};

let dagRendered = false;
function renderDag() {
  if (dagRendered) return;
  dagRendered = true;
  const data = LINEAGE;
  const nodesById = {};
  data.nodes.forEach(n => nodesById[n.id] = n);

  const phases = [];
  data.nodes.forEach(n => { if (!phases.includes(n.phase)) phases.push(n.phase); });
  const phaseRow = {};
  phases.forEach((p, i) => phaseRow[p] = i);

  const rowCounts = {};
  const pos = {};
  data.nodes.forEach(n => {
    const row = phaseRow[n.phase];
    const col = rowCounts[row] || 0;
    rowCounts[row] = col + 1;
    pos[n.id] = { col, row };
  });
  Object.keys(rowCounts).forEach(row => {
    const offset = (rowCounts[row] - 1) / 2;
    Object.keys(pos).forEach(id => {
      if (pos[id].row == row) pos[id].col -= offset;
    });
  });

  const dx = 230, dy = 130, boxW = 200, boxH = 56;
  const maxCol = Math.max(...Object.values(rowCounts));
  const width = (maxCol + 1) * dx + 260;
  const height = phases.length * dy + 80;
  const xy = {};
  Object.keys(pos).forEach(id => {
    xy[id] = { x: pos[id].col * dx + width / 2, y: pos[id].row * dy + 60 };
  });

  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

  // phase labels
  phases.forEach((p, i) => {
    const t = document.createElementNS(svgNS, 'text');
    t.setAttribute('x', 10);
    t.setAttribute('y', i * dy + 60 + 5);
    t.setAttribute('font-size', '13');
    t.setAttribute('font-weight', '700');
    t.setAttribute('fill', 'currentColor');
    t.textContent = p;
    svg.appendChild(t);
  });

  // edges
  data.edges.forEach(e => {
    const a = xy[e.from], b = xy[e.to];
    if (!a || !b) return;
    const style = EDGE_STYLE[e.kind] || EDGE_STYLE.led_to;
    const path = document.createElementNS(svgNS, 'path');
    const midY = (a.y + boxH / 2 + b.y - boxH / 2) / 2;
    const d = `M ${a.x} ${a.y + boxH / 2} C ${a.x} ${midY}, ${b.x} ${midY}, ${b.x} ${b.y - boxH / 2}`;
    path.setAttribute('d', d);
    path.setAttribute('stroke', style.stroke);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke-width', '1.4');
    if (style.dash !== 'none') path.setAttribute('stroke-dasharray', style.dash);
    path.setAttribute('marker-end', 'url(#arrow-' + e.kind + ')');
    svg.appendChild(path);
  });

  // arrow markers
  const defs = document.createElementNS(svgNS, 'defs');
  Object.entries(EDGE_STYLE).forEach(([kind, style]) => {
    const marker = document.createElementNS(svgNS, 'marker');
    marker.setAttribute('id', 'arrow-' + kind);
    marker.setAttribute('viewBox', '0 0 10 10');
    marker.setAttribute('refX', '9');
    marker.setAttribute('refY', '5');
    marker.setAttribute('markerWidth', '7');
    marker.setAttribute('markerHeight', '7');
    marker.setAttribute('orient', 'auto-start-reverse');
    const p = document.createElementNS(svgNS, 'path');
    p.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
    p.setAttribute('fill', style.stroke);
    marker.appendChild(p);
    defs.appendChild(marker);
  });
  svg.appendChild(defs);

  // nodes
  data.nodes.forEach(n => {
    const p = xy[n.id];
    const g = document.createElementNS(svgNS, 'g');
    g.setAttribute('transform', `translate(${p.x - boxW / 2}, ${p.y - boxH / 2})`);
    const rect = document.createElementNS(svgNS, 'rect');
    rect.setAttribute('width', boxW);
    rect.setAttribute('height', boxH);
    rect.setAttribute('rx', 6);
    rect.setAttribute('fill', 'var(--panel)');
    rect.setAttribute('stroke', VERDICT_COLOR[n.verdict] || '#888');
    rect.setAttribute('stroke-width', '2.2');
    rect.classList.add('node-rect');
    rect.addEventListener('click', () => showNodeDetail(n.id));
    g.appendChild(rect);
    const t1 = document.createElementNS(svgNS, 'text');
    t1.setAttribute('x', boxW / 2);
    t1.setAttribute('y', 22);
    t1.setAttribute('text-anchor', 'middle');
    t1.setAttribute('font-size', '10.5');
    t1.setAttribute('fill', 'currentColor');
    t1.style.pointerEvents = 'none';
    t1.textContent = n.id.replace(/_/g, ' ');
    g.appendChild(t1);
    const t2 = document.createElementNS(svgNS, 'text');
    t2.setAttribute('x', boxW / 2);
    t2.setAttribute('y', 40);
    t2.setAttribute('text-anchor', 'middle');
    t2.setAttribute('font-size', '10.5');
    t2.setAttribute('font-weight', '700');
    t2.setAttribute('fill', VERDICT_COLOR[n.verdict] || '#888');
    t2.style.pointerEvents = 'none';
    t2.textContent = n.verdict.toUpperCase();
    g.appendChild(t2);
    svg.appendChild(g);
  });

  const host = document.getElementById('daghost');
  host.innerHTML = '';
  host.appendChild(svg);
  showNodeDetail(data.nodes[0].id);
}

function showNodeDetail(id) {
  const n = LINEAGE.nodes.find(x => x.id === id);
  const panel = document.getElementById('dagpanel');
  const color = VERDICT_COLOR[n.verdict] || '#888';
  panel.innerHTML = `
    <h3>${n.id.replace(/_/g, ' ')}</h3>
    <p><strong>phase:</strong> ${n.phase}</p>
    <p><strong>verdict:</strong> <span style="color:${color};font-weight:700">${n.verdict.toUpperCase()}</span></p>
    <p><strong>hypothesis:</strong> ${n.hypothesis}</p>
    <p><strong>change:</strong> ${n.change}</p>
    <p><strong>judgment number:</strong> ${n.judgment_number || '—'}</p>
    <p><strong>gate:</strong> ${n.gate || '—'}</p>
    <p><strong>source files:</strong><br>${n.source_files.map(s => '<code>' + s + '</code>').join('<br>')}</p>
  `;
}

document.addEventListener('DOMContentLoaded', () => showTab('overview'));
"""


def build():
    lineage_json = json.dumps(LINEAGE)
    provenance_json = json.dumps(PROVENANCE)

    warnings_html = "".join(f"<div>{esc(w)}</div>" for w in PROVENANCE["meta"]["warnings"])

    body = f"""
<header class="top">
git_sha={esc(PROVENANCE['meta']['git_sha_head'][:12])} &nbsp;·&nbsp;
evalphys=v{esc(PROVENANCE['meta']['evalphys_version'])} (manifest sha {esc(PROVENANCE['meta']['git_sha_metrics_manifest'][:12])}) &nbsp;·&nbsp;
generated={esc(PROVENANCE['meta']['generated'])} &nbsp;·&nbsp;
<b>artifacts frozen — no re-scoring</b>
</header>
<nav class="tabs">
  <button id="btn-overview" onclick="showTab('overview')">Overview</button>
  <button id="btn-lineage" onclick="showTab('lineage')">Lineage</button>
  <button id="btn-matrix" onclick="showTab('matrix')">Matrix</button>
  <button id="btn-calibration" onclick="showTab('calibration')">Calibration strata</button>
  <button id="btn-osse" onclick="showTab('osse')">OSSE</button>
  <button id="btn-provenance" onclick="showTab('provenance')">Provenance</button>
</nav>
<main>

<section id="tab-overview" class="tab">
  <h1>NeSPReSO v2 — project evolution</h1>
  <div class="warnbanner">{warnings_html}</div>
  <h2>Abstract</h2>
  <p>This branch built a frozen evaluation standard (evalphys v1.1.0), used it to run a pre-registered
  3-representation × 3-head ablation matrix, and closed the loop with a toy observing-system-simulation
  experiment (OSSE). The matrix's mechanical winner (A×CRPS) clears its probabilistic calibration gate in
  PC space but <b>fails</b> that same gate in physical temperature space (ENCE(T)=0.236). The dissertation's
  central data-assimilation question is answered in the negative: NeSPReSO casts tie an ISOP/MODAS-class
  baseline on fixed R (E3 vs E2: 0.5454 vs 0.5410, FAIL), and a structured calibrated observation-error
  covariance <b>degrades</b> the analysis relative to a diagonal one (0.6160 vs 0.5463), with the cause
  isolated to basis-induced off-diagonal correlation in the CRPS head. Full text: see
  <code>EVOLUTION.md</code> in this directory.</p>
  <h2>Timeline</h2>
  {timeline_table()}
</section>

<section id="tab-lineage" class="tab">
  <h2>Experiment lineage (interactive — click a node)</h2>
  <div style="margin:8px 0 14px">
    <span class="legend-item"><span class="legend-swatch" style="border-color:#2a7f3f"></span>survivor</span>
    <span class="legend-item"><span class="legend-swatch" style="border-color:#b3261e"></span>killed</span>
    <span class="legend-item"><span class="legend-swatch" style="border-color:#b8860b"></span>superseded</span>
    <span class="legend-item"><span class="legend-swatch" style="border-color:#5a4fcf"></span>erratum</span>
  </div>
  <div id="dagwrap">
    <div id="daghost"></div>
    <div id="dagpanel"></div>
  </div>
  <p style="margin-top:16px">Static reference render: {inline_svg('lineage_dag')}</p>
</section>

<section id="tab-matrix" class="tab">
  <h2>Phase 5 matrix</h2>
  <h3>Latent-space judgment table (per-cell judged in each representation's own basis)</h3>
  <p>PC-space CRPS/NLL is <b>not comparable</b> across representations (A: T/S PCA-16, B: joint EOF-32,
  C: density+spice PCA). Only the det (T RMSE) column is physical-space already.</p>
  {matrix_latent_table()}
  {inline_svg('matrix_gate_heatmap')}
  <h3>Physical-space table (comparable across representations; reports/ablation_summary.md)</h3>
  {matrix_physical_table()}
  <p><b>B</b> won deterministic RMSE (0.534). <b>A×CRPS</b> won the probabilistic crown (mechanical
  decision rule). <b>C</b> lost both axes — no C cell clears physical ENCE, and C×det misses the skill floor.</p>
</section>

<section id="tab-calibration" class="tab">
  <h2>Calibration reality — A×CRPS winner, physical T space</h2>
  <p>Overall physical ENCE(T) = 0.2362 ± 0.0053 — <span class="v-fail">FAIL</span> vs the 0.20 gate that
  this same cell passed in PC space (0.053). Source: <code>reports/phase5_A_CRPS_physical_strata.md</code>.</p>
  <div class="figgrid">
    {inline_svg('depthband_season_crps')}
    {inline_svg('depthband_season_ence')}
  </div>
</section>

<section id="tab-osse" class="tab">
  <h2>Phase 6 OSSE — canonical result</h2>
  {osse_table()}
  <p><span class="v-fail">E3 &gt; E2: FAIL</span> (0.5454 not &lt; 0.5410) &nbsp;·&nbsp;
     <span class="v-fail">E4 ≥ E3: FAIL</span> (0.6160 &gt; 0.5454)</p>
  {inline_svg('osse_panel')}

  <h3>Structured-covariance strand — diagonal control</h3>
  {rcal_diag_table()}
  <p>diag(Σ) is exactly preserved by the Schur localization, so the entire 0.5463→0.6160 degradation is
  the CRPS-head off-diagonals alone.</p>
  {inline_svg('diag_control_headline')}
  {inline_svg('rcal_etable_depthband')}
  {inline_svg('rcal_schematic')}

  <h3>Gate-floor ruler repair</h3>
  {inline_svg('ruler_sparkline')}

  <h3>Datacube strand</h3>
  <p>Data-quality and extraction-geometry fix; NOT evidence the patch/residual branch won — it lost the
  Phase 5 matrix on both axes above. The dissertation-winning model (A×CRPS) is a point model.</p>
  <div class="figgrid">
    {inline_svg('cube_schematic')}
    {inline_svg('cube_extraction_inset')}
    {inline_svg('cube_dataflow')}
    {inline_svg('cube_stale_fingerprint')}
  </div>
</section>

<section id="tab-provenance" class="tab">
  <h2>Provenance</h2>
  <h3>Contradictions surfaced (not silently resolved)</h3>
  {contradictions_block()}
  <h3>Requested-but-unsourced numbers (omitted from every figure/table above)</h3>
  {unsourced_block()}
  <h3>Every figure and table, with source</h3>
  {provenance_table()}
</section>

</main>
<footer>Built from frozen artifacts only. reports/evolution/build_html.py · reports/evolution/check_provenance.py</footer>
<script id="lineage-data" type="application/json">{lineage_json}</script>
<script id="provenance-data" type="application/json">{provenance_json}</script>
<script>
const LINEAGE = JSON.parse(document.getElementById('lineage-data').textContent);
const PROVENANCE = JSON.parse(document.getElementById('provenance-data').textContent);
{JS}
</script>
"""

    out = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NeSPReSO v2 — project evolution</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    (HERE / "index.html").write_text(out)
    print(f"wrote {HERE / 'index.html'} ({len(out)/1024:.0f} KB)")


if __name__ == "__main__":
    build()
