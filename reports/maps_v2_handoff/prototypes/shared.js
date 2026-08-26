/* ponytail: throwaway shared deck for layout bakeoff. Not production. */
window.MAPS_PROTO = {
  colors: {
    encoder: "#0B6E4F",
    uncertainty: "#C81D25",
    heave: "#3D348B",
    da: "#E09F3E",
    ink: "#1A1A1A",
    mute: "#5C6570",
    paper: "#F7F5F0",
    line: "#D9D3C7",
  },
  defaultDeck: {
    title: "NeSPReSO v2 → DA handoff",
    slides: [
      {
        id: "backbone",
        title: "SatEncoder",
        subtitle: "Shared backbone (code: PatchConvMLP)",
        body: "Scalar encodings (time, lat, lon, climate indices) plus satellite Sea Surface Temperature (SST), Sea Surface Salinity (SSS), and Sea Surface Height (SSH) enter a patch-aware encoder. Point mode uses a linear satellite embed. Patch mode reshapes the flat satellite block to (B, C, T, H, W) and runs a small Conv2d trunk, then an Multi-Layer Perceptron (MLP) head.",
        equation: "h = EncProj(e) + SatProj(sat)",
        diagram: "encoder",
        accent: "encoder",
      },
      {
        id: "uncertainty",
        title: "Uncertainty Head",
        subtitle: "A×CRPS cell (code: probabilistic PatchConvMLP + PCAHeteroLoss / DensitySpiceProbLoss)",
        body: "The head emits mean μ and scale σ per Principal Component Analysis (PCA) score (or density–spice target). Training uses the closed-form Continuous Ranked Probability Score (CRPS) for a Gaussian forecast. Softplus keeps σ > σ_min.",
        equation: "CRPS(μ,σ,y) = σ [ z(2Φ(z)−1) + 2φ(z) − 1/√π ],  z=(y−μ)/σ",
        diagram: "uncertainty",
        accent: "uncertainty",
      },
      {
        id: "heave",
        title: "Heave Residual",
        subtitle: "HeaveResidualFast (code: HeaveResidual + batched warp)",
        body: "Predict Mixed Layer Depth (MLD), depth of the 26 °C isotherm (D26), and stretch, plus residual Temperature/Salinity (T/S) PCs on a canonical depth grid. Decode = warp climatology to the predicted landmarks, add residual, unwarp to physical depth. Fast path vectorizes the same searchsorted lerp.",
        equation: "T_phys = Unwarp( Warp(T_clim; MLD, D26) + V z_T + μ_T )",
        diagram: "heave",
        accent: "heave",
      },
      {
        id: "da",
        title: "DA handoff",
        subtitle: "What Data Assimilation (DA) consumes",
        body: "Export a profile mean and a usable observation-error scale. Prefer diagonal σ in physical space when off-diagonal PCA-induced covariances hurt column Optimal Interpolation (OI). Contract fields: μ(z), σ(z) or calibrated σ_o, depth grid, and provenance (cache + checkpoint pair).",
        equation: "y^o = μ(z),   R ≈ diag(σ_o(z)^2)",
        diagram: "da",
        accent: "da",
      },
    ],
  },
};

window.MAPS_PROTO.renderDiagram = function (kind, el) {
  const C = window.MAPS_PROTO.colors;
  const svg = {
    encoder: `
      <svg viewBox="0 0 420 160" width="100%" aria-hidden="true">
        <rect x="8" y="50" width="70" height="60" rx="8" fill="${C.encoder}" opacity=".15" stroke="${C.encoder}" stroke-width="2"/>
        <text x="43" y="85" text-anchor="middle" font-size="11" fill="${C.encoder}">enc</text>
        <rect x="100" y="40" width="90" height="80" rx="8" fill="${C.encoder}" opacity=".2" stroke="${C.encoder}" stroke-width="2"/>
        <text x="145" y="75" text-anchor="middle" font-size="11" fill="${C.encoder}">sat patch</text>
        <text x="145" y="92" text-anchor="middle" font-size="10" fill="${C.mute}">Conv2d</text>
        <path d="M190 80 H220" stroke="${C.ink}" stroke-width="2" marker-end="url(#arrow)"/>
        <rect x="230" y="45" width="100" height="70" rx="8" fill="${C.encoder}" stroke="${C.encoder}" stroke-width="2"/>
        <text x="280" y="75" text-anchor="middle" font-size="12" fill="#fff" font-weight="700">SatEncoder</text>
        <text x="280" y="94" text-anchor="middle" font-size="10" fill="#e8fff4">h ∈ R^d</text>
        <path d="M330 80 H360" stroke="${C.ink}" stroke-width="2"/>
        <rect x="360" y="55" width="50" height="50" rx="8" fill="${C.paper}" stroke="${C.line}" stroke-width="2"/>
        <text x="385" y="85" text-anchor="middle" font-size="11" fill="${C.mute}">head</text>
      </svg>`,
    uncertainty: `
      <svg viewBox="0 0 420 160" width="100%" aria-hidden="true">
        <rect x="20" y="45" width="90" height="70" rx="8" fill="${C.encoder}" stroke="${C.encoder}" stroke-width="2"/>
        <text x="65" y="85" text-anchor="middle" font-size="12" fill="#fff" font-weight="700">h</text>
        <path d="M110 80 H150" stroke="${C.ink}" stroke-width="2"/>
        <rect x="150" y="20" width="100" height="50" rx="8" fill="${C.uncertainty}" opacity=".2" stroke="${C.uncertainty}" stroke-width="2"/>
        <text x="200" y="50" text-anchor="middle" font-size="12" fill="${C.uncertainty}" font-weight="700">μ_out</text>
        <rect x="150" y="90" width="100" height="50" rx="8" fill="${C.uncertainty}" stroke="${C.uncertainty}" stroke-width="2"/>
        <text x="200" y="120" text-anchor="middle" font-size="12" fill="#fff" font-weight="700">σ_out</text>
        <path d="M250 45 H290 V80 H310" stroke="${C.uncertainty}" stroke-width="2" fill="none"/>
        <path d="M250 115 H290 V80" stroke="${C.uncertainty}" stroke-width="2" fill="none"/>
        <rect x="310" y="50" width="90" height="60" rx="8" fill="${C.uncertainty}" opacity=".15" stroke="${C.uncertainty}" stroke-width="2"/>
        <text x="355" y="78" text-anchor="middle" font-size="11" fill="${C.uncertainty}">[μ, σ]</text>
        <text x="355" y="96" text-anchor="middle" font-size="10" fill="${C.mute}">CRPS train</text>
      </svg>`,
    heave: `
      <svg viewBox="0 0 420 170" width="100%" aria-hidden="true">
        <rect x="10" y="20" width="80" height="40" rx="6" fill="${C.heave}" opacity=".2" stroke="${C.heave}" stroke-width="2"/>
        <text x="50" y="45" text-anchor="middle" font-size="11" fill="${C.heave}">warp 3</text>
        <rect x="10" y="70" width="80" height="40" rx="6" fill="${C.heave}" stroke="${C.heave}" stroke-width="2"/>
        <text x="50" y="95" text-anchor="middle" font-size="11" fill="#fff">T/S PCs</text>
        <path d="M90 85 H120" stroke="${C.ink}" stroke-width="2"/>
        <rect x="120" y="30" width="110" height="110" rx="8" fill="${C.paper}" stroke="${C.heave}" stroke-width="2"/>
        <text x="175" y="60" text-anchor="middle" font-size="11" fill="${C.heave}" font-weight="700">canonical z</text>
        <text x="175" y="82" text-anchor="middle" font-size="10" fill="${C.mute}">MLD₀=50 m</text>
        <text x="175" y="98" text-anchor="middle" font-size="10" fill="${C.mute}">D26₀=120 m</text>
        <text x="175" y="120" text-anchor="middle" font-size="10" fill="${C.mute}">+ residual</text>
        <path d="M230 85 H260" stroke="${C.ink}" stroke-width="2"/>
        <rect x="260" y="45" width="140" height="80" rx="8" fill="${C.heave}" opacity=".15" stroke="${C.heave}" stroke-width="2"/>
        <text x="330" y="80" text-anchor="middle" font-size="12" fill="${C.heave}" font-weight="700">unwarp → T,S(z)</text>
        <text x="330" y="100" text-anchor="middle" font-size="10" fill="${C.mute}">HeaveResidualFast</text>
      </svg>`,
    da: `
      <svg viewBox="0 0 420 160" width="100%" aria-hidden="true">
        <rect x="20" y="40" width="120" height="80" rx="8" fill="${C.uncertainty}" opacity=".15" stroke="${C.uncertainty}" stroke-width="2"/>
        <text x="80" y="75" text-anchor="middle" font-size="12" fill="${C.uncertainty}" font-weight="700">μ(z), σ(z)</text>
        <text x="80" y="95" text-anchor="middle" font-size="10" fill="${C.mute}">model export</text>
        <path d="M140 80 H180" stroke="${C.da}" stroke-width="3"/>
        <rect x="180" y="35" width="100" height="90" rx="8" fill="${C.da}" stroke="${C.da}" stroke-width="2"/>
        <text x="230" y="75" text-anchor="middle" font-size="13" fill="#1a1200" font-weight="700">handoff</text>
        <text x="230" y="95" text-anchor="middle" font-size="10" fill="#3a2a00">diag R preferred</text>
        <path d="M280 80 H320" stroke="${C.da}" stroke-width="3"/>
        <rect x="320" y="40" width="80" height="80" rx="8" fill="${C.paper}" stroke="${C.line}" stroke-width="2"/>
        <text x="360" y="85" text-anchor="middle" font-size="12" fill="${C.mute}">DA / OI</text>
      </svg>`,
  };
  el.innerHTML = svg[kind] || svg.encoder;
};

window.MAPS_PROTO.bindEditor = function (opts) {
  const { textarea, statusEl, onDeck } = opts;
  let deck = JSON.parse(JSON.stringify(window.MAPS_PROTO.defaultDeck));
  textarea.value = JSON.stringify(deck, null, 2);
  const apply = () => {
    try {
      deck = JSON.parse(textarea.value);
      statusEl.textContent = "JSON ok · live preview";
      statusEl.dataset.ok = "1";
      onDeck(deck);
    } catch (err) {
      statusEl.textContent = "JSON error: " + err.message;
      statusEl.dataset.ok = "0";
    }
  };
  textarea.addEventListener("input", apply);
  apply();
  return { getDeck: () => deck, apply };
};
