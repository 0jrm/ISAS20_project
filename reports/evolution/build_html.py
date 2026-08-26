#!/usr/bin/env python3
"""Build reports/evolution/index.html — single self-contained file.

Embeds lineage.json, PROVENANCE.json summary tables, and all generated SVGs
inline. No network, no server, no localStorage.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
ROOT = HERE.parent.parent


def svg_inline(name: str) -> str:
    raw = (FIGS / f"{name}.svg").read_text()
    # strip XML declaration for embedding
    if raw.startswith("<?xml"):
        raw = raw.split("?>", 1)[1].lstrip()
    return raw


def png_data_uri(rel: str) -> str:
    data = (ROOT / rel).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main() -> None:
    provenance = json.loads((HERE / "PROVENANCE.json").read_text())
    lineage = json.loads((HERE / "lineage.json").read_text())
    meta = provenance["meta"]
    head = meta["git_sha_head"]
    evalphys = meta["git_sha_metrics_manifest"]
    generated = meta["generated"]
    eval_ver = meta["evalphys_version"]

    lineage_js = json.dumps(lineage)
    provenance_js = json.dumps(provenance)

    svgs = {
        "lineage_dag": svg_inline("lineage_dag"),
        "matrix_gate_heatmap": svg_inline("matrix_gate_heatmap"),
        "depthband_season_crps": svg_inline("depthband_season_crps"),
        "depthband_season_ence": svg_inline("depthband_season_ence"),
        "osse_panel": svg_inline("osse_panel"),
        "ruler_sparkline": svg_inline("ruler_sparkline"),
    }

    reused = {
        "depth_rmse_bias": png_data_uri(
            "NeSPReSO2_onTemplate/notebooks/compare_outputs/depth_rmse_bias.png"
        ),
        "argo_production_depth_rmse": png_data_uri(
            "NeSPReSO2_onTemplate/notebooks/compare_outputs/argo_production_depth_rmse.png"
        ),
        "depth_rmse_overlay": png_data_uri(
            "NeSPReSO2_onTemplate/notebooks/compare_outputs/depth_rmse_overlay.png"
        ),
    }

    # Summary tables as inline JSON for the Overview tab
    tables = {
        "timeline": [
            {"phase": "0", "question": "Data + split + evalphys freeze", "verdict": "survivor",
             "source": "data_census.md / split_design.md / METRICS_MANIFEST.json"},
            {"phase": "1", "question": "Soft bases fix stability? Stale sat?", "verdict": "soft killed; D+T2 survivor",
             "source": "phase1_decisive_tests.md / stale_by_split.md"},
            {"phase": "3", "question": "Ruler + low-rank density skill", "verdict": "erratum + σ0 admit / a-space killed",
             "source": "gate_floor_provenance.md"},
            {"phase": "4", "question": "Heteroscedastic calibration", "verdict": "v1 killed; s2b ENCE 0.1603 PASS",
             "source": "phase4_ence_recalib_s2b.md"},
            {"phase": "5", "question": "3x3 matrix + strata", "verdict": "A×CRPS winner; strata ENCE(T) FAIL",
             "source": "ablation_summary.md / phase5_A_CRPS_physical_strata.md"},
            {"phase": "6", "question": "OSSE E3>E2 / E4≥E3", "verdict": "FAIL / FAIL",
             "source": "osse_results.md"},
        ],
        "matrix": {
            "B_det_T": 0.534,
            "A_CRPS_phys_CRPS": 0.119,
            "A_CRPS_phys_ENCE": 0.153,
            "A_CRPS_T": 0.559,
            "C_det_T": 0.609,
            "floor": 0.5903,
        },
        "osse": {
            "E2": 0.5410,
            "E3": 0.5454,
            "E4": 0.6160,
            "E5": 1.4008,
            "E3_gt_E2": "FAIL",
            "E4_ge_E3": "FAIL",
            "diag_control": 0.546,
        },
        "calibration": {
            "pc_ENCE": 0.053,
            "phys_ENCE_TS": 0.153,
            "phys_ENCE_T": 0.2362,
        },
    }
    tables_js = json.dumps(tables)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NeSPReSO project evolution (frozen artifacts)</title>
<style>
:root {{
  --bg: #f6f4ef;
  --ink: #1c1a16;
  --muted: #5a564c;
  --card: #fffcf7;
  --line: #d4cfc3;
  --pass: #2a7f3f;
  --fail: #b3261e;
  --warn: #8a5a00;
  --accent: #1a5276;
  --tab: #e8e2d6;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; font-family: "IBM Plex Sans", "Source Sans 3", "Segoe UI", sans-serif;
  background: var(--bg); color: var(--ink); line-height: 1.45;
}}
.banner {{
  position: sticky; top: 0; z-index: 20;
  background: #1c1a16; color: #f0ebe3; padding: 0.55rem 1rem;
  font-family: "IBM Plex Mono", "Consolas", monospace; font-size: 0.78rem;
  display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; align-items: center;
}}
.banner .tag {{ color: #c9b896; }}
.warnbar {{
  background: #5c3b00; color: #ffe9b8; padding: 0.55rem 1rem; font-size: 0.85rem;
  border-bottom: 2px solid #c47a00;
}}
.warnbar + .warnbar {{ border-top: 1px solid #8a5a00; }}
header.hero {{
  padding: 1.5rem 1.25rem 0.5rem; max-width: 1100px; margin: 0 auto;
}}
header.hero h1 {{ margin: 0 0 0.35rem; font-size: 1.65rem; letter-spacing: -0.02em; }}
header.hero p {{ margin: 0; color: var(--muted); max-width: 62ch; }}
nav.tabs {{
  display: flex; flex-wrap: wrap; gap: 0.35rem; padding: 0.75rem 1.25rem;
  max-width: 1100px; margin: 0 auto; border-bottom: 1px solid var(--line);
  position: sticky; top: 2.1rem; background: var(--bg); z-index: 15;
}}
nav.tabs button {{
  border: 1px solid var(--line); background: var(--tab); color: var(--ink);
  padding: 0.4rem 0.75rem; border-radius: 4px; cursor: pointer; font-size: 0.9rem;
}}
nav.tabs button.active {{ background: var(--accent); color: white; border-color: var(--accent); }}
main {{ max-width: 1100px; margin: 0 auto; padding: 1rem 1.25rem 3rem; }}
section.panel {{ display: none; }}
section.panel.active {{ display: block; }}
h2 {{ font-size: 1.25rem; margin: 0.5rem 0 0.75rem; }}
h3 {{ font-size: 1.05rem; margin: 1.25rem 0 0.5rem; }}
table {{
  width: 100%; border-collapse: collapse; background: var(--card);
  font-size: 0.9rem; margin: 0.75rem 0 1.25rem;
}}
th, td {{ border: 1px solid var(--line); padding: 0.4rem 0.55rem; text-align: left; vertical-align: top; }}
th {{ background: #efe9dc; }}
.fail {{ color: var(--fail); font-weight: 700; }}
.pass {{ color: var(--pass); font-weight: 700; }}
.fig {{
  background: var(--card); border: 1px solid var(--line); padding: 0.75rem;
  margin: 0.75rem 0 1.25rem; overflow-x: auto;
}}
.fig svg {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
.fig img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
.note {{ color: var(--muted); font-size: 0.88rem; }}
#node-detail {{
  background: var(--card); border: 1px solid var(--line); padding: 0.75rem 1rem;
  min-height: 5rem; margin-top: 0.75rem;
}}
#node-detail dl {{ display: grid; grid-template-columns: 8rem 1fr; gap: 0.25rem 0.75rem; margin: 0; }}
#node-detail dt {{ color: var(--muted); }}
.clickable-nodes {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 0.5rem; }}
.node-chips {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.5rem 0 1rem; }}
.node-chips button {{
  font-size: 0.75rem; padding: 0.25rem 0.45rem; border-radius: 3px; cursor: pointer;
  border: 1px solid #333; color: white;
}}
.v-survivor {{ background: var(--pass); }}
.v-killed {{ background: var(--fail); }}
.v-superseded {{ background: #6b5b95; }}
.v-erratum {{ background: #c47a00; }}
code {{ font-family: "IBM Plex Mono", Consolas, monospace; font-size: 0.84em; }}
</style>
</head>
<body>
<div class="banner">
  <span><span class="tag">git_sha=</span>{head}</span>
  <span><span class="tag">evalphys=</span>{eval_ver} ({evalphys[:12]}…)</span>
  <span><span class="tag">generated=</span>{generated}</span>
  <span><span class="tag">artifacts frozen</span> — no re-scoring</span>
</div>
<div class="warnbar">
  Warning: METRICS_MANIFEST git_sha ({evalphys}) ≠ HEAD ({head}). Metric definitions unchanged; manifest sha is stale relative to HEAD.
</div>
<div class="warnbar">
  Warning: phase6_osse/run.log E4=2.4186 / E5=1.8906 disagrees with cast_column_s42.json + osse_results.md (E4=0.6160 / E5=1.4008). Displayed numbers use the agreeing JSON+MD pair only.
</div>

<header class="hero">
  <h1>NeSPReSO project evolution</h1>
  <p>Provenance-traced reconstruction from frozen reports. No model was run, retrained, or re-scored to produce this page.</p>
</header>

<nav class="tabs" role="tablist">
  <button type="button" class="active" data-tab="overview">Overview</button>
  <button type="button" data-tab="lineage">Lineage</button>
  <button type="button" data-tab="matrix">Matrix</button>
  <button type="button" data-tab="strata">Calibration strata</button>
  <button type="button" data-tab="osse">OSSE</button>
  <button type="button" data-tab="provenance">Provenance</button>
</nav>

<main>
<section id="overview" class="panel active">
  <h2>Overview</h2>
  <p>B won deterministic RMSE (<strong>0.534</strong>). A won the probabilistic crown (A×CRPS: phys CRPS <strong>0.119</strong>, phys ENCE <strong>0.153</strong>). C lost both (det <strong>0.609</strong>; phys ENCE ≥ <strong>0.384</strong>). Physical ENCE(T) strata overall <strong>0.2362</strong> — FAIL. OSSE: NeSPReSO ties ISOP (<strong>0.5454</strong> vs <strong>0.5410</strong>); E3&gt;E2 <span class="fail">FAIL</span>; E4≥E3 <span class="fail">FAIL</span> (E4=<strong>0.6160</strong>).</p>

  <h3>Timeline</h3>
  <table id="tbl-timeline"><thead><tr><th>phase</th><th>question</th><th>verdict</th><th>source</th></tr></thead><tbody></tbody></table>

  <h3>Ruler repair</h3>
  <div class="fig">{svgs["ruler_sparkline"]}</div>

  <h3>Reused diagnostic PNGs (not regenerated)</h3>
  <div class="fig"><img alt="depth RMSE bias" src="{reused["depth_rmse_bias"]}"/></div>
  <div class="fig"><img alt="argo production depth RMSE" src="{reused["argo_production_depth_rmse"]}"/></div>
  <div class="fig"><img alt="depth RMSE overlay" src="{reused["depth_rmse_overlay"]}"/></div>
</section>

<section id="lineage" class="panel">
  <h2>Lineage</h2>
  <p class="clickable-nodes">Click a node chip to inspect hypothesis / change / verdict / source. DAG layout is presentational; content is from <code>lineage.json</code>.</p>
  <div class="node-chips" id="node-chips"></div>
  <div class="fig">{svgs["lineage_dag"]}</div>
  <div id="node-detail"><em>Select a node.</em></div>
</section>

<section id="matrix" class="panel">
  <h2>Matrix</h2>
  <p>Latent judgment numbers with PASS/FAIL glyphs. Hatched cells = PC-space CRPS (not comparable across representations). Ring = Section 3 winner A×CRPS.</p>
  <div class="fig">{svgs["matrix_gate_heatmap"]}</div>
  <table>
    <thead><tr><th>claim</th><th>number</th><th>source</th></tr></thead>
    <tbody>
      <tr><td>B deterministic RMSE</td><td><strong>0.534</strong> ≤ 0.5903</td><td>phase5_B_det.md</td></tr>
      <tr><td>A×CRPS phys CRPS / ENCE / T</td><td><strong>0.119</strong> / <strong>0.153</strong> / <strong>0.559</strong></td><td>ablation_summary.md</td></tr>
      <tr><td>C det T</td><td class="fail">0.609</td><td>phase5_C_det.md</td></tr>
    </tbody>
  </table>
</section>

<section id="strata" class="panel">
  <h2>Calibration strata</h2>
  <p>PC-space ENCE <span class="pass">0.053 PASS</span>; pooled phys ENCE(T+S) <span class="pass">0.153 PASS</span>; physical ENCE(T) overall <span class="fail">0.2362 FAIL</span>. Fail borders at ENCE ≥ 0.20.</p>
  <div class="fig">{svgs["depthband_season_crps"]}</div>
  <div class="fig">{svgs["depthband_season_ence"]}</div>
</section>

<section id="osse" class="panel">
  <h2>OSSE</h2>
  <p>NeSPReSO ties ISOP. Calibrated full-localized R does not beat fixed R; diag-control 0.546 preferred over full 0.616.</p>
  <p><span class="fail">E3&gt;E2 FAIL</span> (0.5454 vs 0.5410) · <span class="fail">E4≥E3 FAIL</span> (0.6160 vs 0.5454)</p>
  <div class="fig">{svgs["osse_panel"]}</div>
</section>

<section id="provenance" class="panel">
  <h2>Provenance</h2>
  <p class="note">Every figure and table rendered in this report. Source paths and git_sha are visible below. <code>unsourced[]</code> is empty.</p>
  <table id="tbl-prov">
    <thead><tr><th>artifact_id</th><th>kind</th><th>path</th><th>git_sha</th><th>judgment_number</th><th>gate_verdict</th><th>source_files</th><th>caveats</th></tr></thead>
    <tbody></tbody>
  </table>
  <h3>Contradictions surfaced</h3>
  <table id="tbl-contra">
    <thead><tr><th>id</th><th>description</th><th>handling</th></tr></thead>
    <tbody></tbody>
  </table>
</section>
</main>

<script>
const LINEAGE = {lineage_js};
const PROVENANCE = {provenance_js};
const TABLES = {tables_js};

// Tabs
document.querySelectorAll('nav.tabs button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('nav.tabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('section.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  }});
}});

// Timeline table
(() => {{
  const tb = document.querySelector('#tbl-timeline tbody');
  TABLES.timeline.forEach(r => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{r.phase}}</td><td>${{r.question}}</td><td>${{r.verdict}}</td><td><code>${{r.source}}</code></td>`;
    tb.appendChild(tr);
  }});
}})();

// Lineage node chips + detail
(() => {{
  const chips = document.getElementById('node-chips');
  const detail = document.getElementById('node-detail');
  function show(n) {{
    const sources = (n.source_files || []).map(s => `<code>${{s}}</code>`).join('<br/>');
    const vclass = 'fail';
    const vhtml = n.verdict === 'survivor' || n.verdict === 'superseded'
      ? `<span class="pass">${{n.verdict}}</span>`
      : `<span class="fail">${{n.verdict}}</span>`;
    // FAIL stays red even for flagship-adjacent killed nodes
    detail.innerHTML = `<dl>
      <dt>id</dt><dd><code>${{n.id}}</code></dd>
      <dt>phase</dt><dd>${{n.phase}}</dd>
      <dt>hypothesis</dt><dd>${{n.hypothesis}}</dd>
      <dt>change</dt><dd>${{n.change}}</dd>
      <dt>judgment</dt><dd>${{n.judgment_number ?? '—'}}</dd>
      <dt>gate</dt><dd>${{n.gate ?? '—'}}</dd>
      <dt>verdict</dt><dd>${{vhtml}}</dd>
      <dt>sources</dt><dd>${{sources}}</dd>
    </dl>`;
  }}
  LINEAGE.nodes.forEach(n => {{
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = n.id;
    b.className = 'v-' + n.verdict;
    b.addEventListener('click', () => show(n));
    chips.appendChild(b);
  }});
}})();

// Provenance table
(() => {{
  const tb = document.querySelector('#tbl-prov tbody');
  PROVENANCE.artifacts.forEach(a => {{
    const tr = document.createElement('tr');
    const path = a.path || a.png_path || '—';
    const src = (a.source_files || []).join('<br/>');
    const cav = (a.caveats || []).join('<br/>');
    const gv = a.gate_verdict || '—';
    const gvHtml = /FAIL/.test(String(gv))
      ? `<span class="fail">${{gv}}</span>`
      : (a.gate_verdict ? `<span class="pass">${{gv}}</span>` : '—');
    tr.innerHTML = `<td><code>${{a.artifact_id}}</code></td><td>${{a.kind}}</td>
      <td><code>${{path}}</code></td><td><code>${{a.git_sha.slice(0,12)}}…</code></td>
      <td>${{a.judgment_number ?? '—'}}</td><td>${{gvHtml}}</td>
      <td>${{src}}</td><td>${{cav || '—'}}</td>`;
    tb.appendChild(tr);
  }});
  const tc = document.querySelector('#tbl-contra tbody');
  (PROVENANCE.contradictions || []).forEach(c => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><code>${{c.id}}</code></td><td>${{c.description}}</td><td>${{c.handling}}</td>`;
    tc.appendChild(tr);
  }});
}})();
</script>
</body>
</html>
"""
    out = HERE / "index.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
