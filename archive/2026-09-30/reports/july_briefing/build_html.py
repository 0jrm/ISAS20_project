#!/usr/bin/env python3
"""Build reports/july_briefing/index.html — self-contained advisor briefing."""
from __future__ import annotations

import base64
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"


def svg_inline(name: str) -> str:
    raw = (FIGS / f"{name}.svg").read_text()
    if raw.startswith("<?xml"):
        raw = raw.split("?>", 1)[1].lstrip()
    return raw


def png_data_uri(name: str) -> str:
    data = (FIGS / f"{name}.png").read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def md_to_simple_html(md: str) -> str:
    """Minimal markdown → HTML for this briefing only (stdlib)."""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_table = False
    in_code = False
    code_lang = ""

    def flush_para(buf: list[str]) -> None:
        if not buf:
            return
        text = " ".join(buf)
        out.append(f"<p>{inline(text)}</p>")
        buf.clear()

    def inline(s: str) -> str:
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        return s

    para: list[str] = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_para(para)
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                in_code = True
                code_lang = line[3:].strip()
                cls = "eq" if code_lang == "eq" else (code_lang or "code")
                out.append(f'<pre class="{cls}"><code>')
            i += 1
            continue
        if in_code:
            out.append(line.replace("&", "&amp;").replace("<", "&lt;") + "\n")
            i += 1
            continue

        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:-]+\|", lines[i + 1] or ""):
            flush_para(para)
            # table
            headers = [c.strip() for c in line.strip("|").split("|")]
            i += 2  # skip separator
            out.append("<table><thead><tr>")
            for h in headers:
                out.append(f"<th>{inline(h)}</th>")
            out.append("</tr></thead><tbody>")
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append("<tr>")
                for c in cells:
                    out.append(f"<td>{inline(c)}</td>")
                out.append("</tr>")
                i += 1
            out.append("</tbody></table>")
            continue

        m_img = re.match(r"!\[([^\]]*)\]\((figs/[^)]+)\)", line.strip())
        if m_img:
            flush_para(para)
            alt, path = m_img.group(1), m_img.group(2)
            stem = Path(path).stem
            svg_path = FIGS / f"{stem}.svg"
            if svg_path.exists():
                out.append(f'<figure class="fig"><figcaption>{inline(alt)}</figcaption>{svg_inline(stem)}</figure>')
            else:
                out.append(
                    f'<figure class="fig"><figcaption>{inline(alt)}</figcaption>'
                    f'<img alt="{alt}" src="{png_data_uri(stem)}"/></figure>'
                )
            i += 1
            continue

        if line.startswith("# "):
            flush_para(para)
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_para(para)
            hid = re.sub(r"[^a-z0-9]+", "-", line[3:].lower()).strip("-")
            out.append(f'<h2 id="{hid}">{inline(line[3:])}</h2>')
        elif line.startswith("### "):
            flush_para(para)
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.strip() == "---":
            flush_para(para)
            out.append("<hr/>")
        elif line.startswith("- "):
            flush_para(para)
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        elif line.startswith(tuple(f"{n}. " for n in range(1, 10))):
            flush_para(para)
            out.append("<ol>")
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                out.append(f"<li>{inline(re.sub(r'^\d+\. ', '', lines[i]))}</li>")
                i += 1
            out.append("</ol>")
            continue
        elif line.strip() == "":
            flush_para(para)
        else:
            para.append(line.strip())
        i += 1
    flush_para(para)
    return "\n".join(out)


def main() -> None:
    md = (HERE / "BRIEFING.md").read_text()
    # drop the first H1 from body; page title owns it
    body_md = re.sub(r"^# .+\n+", "", md, count=1)
    body = md_to_simple_html(body_md)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>July briefing — Phase 5 matrix &amp; Phase 6 OSSE</title>
<style>
:root {{
  --bg: #f4f1ea;
  --ink: #1c1914;
  --muted: #5c564c;
  --card: #fffcf7;
  --line: #d6d0c4;
  --accent: #245b4e;
  --warn: #8a4b12;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Source Serif 4", "Iowan Old Style", "Palatino Linotype", Palatino, serif;
  background:
    radial-gradient(1200px 500px at 10% -10%, #e7efe9 0%, transparent 55%),
    radial-gradient(900px 400px at 100% 0%, #efe6d6 0%, transparent 50%),
    var(--bg);
  color: var(--ink);
  line-height: 1.55;
}}
.top {{
  background: var(--accent); color: #f4fff9; padding: 0.65rem 1.25rem;
  font-family: "IBM Plex Sans", "Source Sans 3", sans-serif; font-size: 0.85rem;
}}
header.hero {{
  max-width: 820px; margin: 0 auto; padding: 2rem 1.25rem 1rem;
}}
header.hero h1 {{
  font-size: 2rem; line-height: 1.15; margin: 0 0 0.5rem; letter-spacing: -0.02em;
}}
header.hero .dek {{ color: var(--muted); font-size: 1.05rem; max-width: 40rem; }}
nav.toc {{
  max-width: 820px; margin: 0 auto; padding: 0 1.25rem 1rem;
  font-family: "IBM Plex Sans", "Source Sans 3", sans-serif; font-size: 0.9rem;
}}
nav.toc a {{ color: var(--accent); margin-right: 0.85rem; text-decoration: none; border-bottom: 1px solid transparent; }}
nav.toc a:hover {{ border-bottom-color: var(--accent); }}
main {{
  max-width: 820px; margin: 0 auto; padding: 0 1.25rem 3rem;
}}
h2 {{
  font-size: 1.45rem; margin-top: 2rem; border-top: 1px solid var(--line); padding-top: 1.25rem;
}}
h3 {{ font-size: 1.15rem; margin-top: 1.4rem; }}
p, li {{ font-size: 1.02rem; }}
code, pre {{
  font-family: "IBM Plex Mono", Consolas, monospace; font-size: 0.9em;
}}
pre {{
  background: #1c1914; color: #f4f1ea; padding: 0.85rem 1rem; overflow-x: auto;
  border-radius: 4px;
}}
pre.eq {{
  background: var(--card); color: var(--ink); border: 1px solid var(--line);
  border-left: 3px solid var(--accent); font-size: 1.02rem; line-height: 1.55;
}}
table {{
  width: 100%; border-collapse: collapse; background: var(--card);
  font-family: "IBM Plex Sans", "Source Sans 3", sans-serif; font-size: 0.88rem;
  margin: 1rem 0 1.25rem;
}}
th, td {{ border: 1px solid var(--line); padding: 0.4rem 0.55rem; vertical-align: top; }}
th {{ background: #ebe4d6; text-align: left; }}
figure.fig {{
  background: var(--card); border: 1px solid var(--line); padding: 0.75rem;
  margin: 1rem 0 1.5rem;
}}
figure.fig figcaption {{
  font-family: "IBM Plex Sans", sans-serif; font-size: 0.85rem; color: var(--muted);
  margin-bottom: 0.5rem;
}}
figure.fig svg, figure.fig img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
.math {{
  font-family: "IBM Plex Mono", Consolas, monospace; font-size: 0.92rem;
  background: var(--card); border-left: 3px solid var(--accent);
  padding: 0.75rem 1rem; margin: 1rem 0; overflow-x: auto;
}}
hr {{ border: none; border-top: 1px solid var(--line); margin: 1.5rem 0; }}
a {{ color: var(--accent); }}
.foot {{
  max-width: 820px; margin: 0 auto; padding: 1rem 1.25rem 2rem;
  font-family: "IBM Plex Sans", sans-serif; font-size: 0.85rem; color: var(--muted);
}}
</style>
</head>
<body>
<div class="top">NeSPReSO · residual_cube · July Phase 5–6 briefing · frozen artifacts (no re-scoring)</div>
<header class="hero">
  <h1>July briefing — Phase 5 matrix &amp; Phase 6 OSSE</h1>
  <p class="dek">For the advisor and lab colleagues. Starts from where we were before the July bake-off and assimilation test, then explains A/B/C, the heads, E0–E5, and what the numbers mean.</p>
</header>
<nav class="toc">
  <a href="#0-where-we-were-before-this-update">Before</a>
  <a href="#1-what-the-system-does-one-picture">System</a>
  <a href="#1-5-the-datacube-where-surface-inputs-come-from">Datacube</a>
  <a href="#2-flavors-a-b-and-c-what-actually-differs">A/B/C</a>
  <a href="#3-the-three-heads-what-the-network-is-asked-to-learn">Heads (CRPS/NLL/ENCE)</a>
  <a href="#5-matrix-result-who-won-what">Matrix</a>
  <a href="#6-is-the-uncertainty-good-where-it-matters">Calibration</a>
  <a href="#7-osse-e0-through-e5">OSSE</a>
  <a href="#9-takeaways-for-discussion">Takeaways</a>
</nav>
<main>
{body}
</main>
<div class="foot">
  Companion provenance ledger: <code>reports/evolution/</code>.
  Rebuild figures: <code>conda run -n nespreso python3 reports/july_briefing/make_figures.py</code>
  then <code>python3 reports/july_briefing/build_html.py</code>.
</div>
</body>
</html>
"""
    out = HERE / "index.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
