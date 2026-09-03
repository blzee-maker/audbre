"""Generate the architecture diagram as SVG, once per colour scheme.

GitHub swaps the two with <picture>/prefers-color-scheme. Keeping both from one
generator is what stops them drifting apart. Run after editing:

    python design/make_diagram.py
"""
from __future__ import annotations

from pathlib import Path

W, H = 1120, 390
OUT = Path(__file__).resolve().parent

# --- geometry -------------------------------------------------------------

BOX_Y, BOX_H, HEAD_H = 70, 210, 27
BOXES = [
    (30,  230, "BROWSER", [
        "waveform, with a grey",
        "ghost of the original",
        "drag to mark a span",
        "A/B audition",
        "removal stack",
    ]),
    (400, 280, "FASTAPI", [
        "ffmpeg decode, any format",
        "30s windows, 2s overlap",
        "cross-faded overlap-add",
        "layer deltas on disk",
        "ffmpeg remux on export",
    ]),
    (840, 250, "MODAL · A10G", [
        "sam-audio-large",
        "flow-matching",
        "transformer",
        "",
        "warm 4 min between edits",
    ]),
]

OUT_Y, BACK_Y = 125, 225
ARROWS = [
    (262, 398, OUT_Y,  "→", 330, "file · prompt · span"),
    (398, 262, BACK_Y, "←", 330, "peaks · preview audio"),
    (682, 838, OUT_Y,  "→", 760, "30s WAV chunk + anchors"),
    (838, 682, BACK_Y, "←", 760, "target + residual"),
]

# The boundary breaks where the arrows cross it - the gap is the point.
DIVIDER_X = 760
DIVIDER_SEGMENTS = [(45, 96), (262, 334)]  # stops above the caption rule

CAPTION = "current  =  base  −  Σ (delta for every enabled layer)"

SCHEMES = {
    "light": {"ink": "#000000", "dim": "#767676", "faint": "#C9C9C9", "paper": "#FFFFFF"},
    "dark":  {"ink": "#E8E8E8", "dim": "#9A9A9A", "faint": "#4A4A4A", "paper": "#0D1117"},
}

MONO = "'Courier New', Courier, monospace"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(p: dict[str, str]) -> str:
    ink, dim, faint, paper = p["ink"], p["dim"], p["faint"], p["paper"]
    o: list[str] = []
    add = o.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="{MONO}" role="img" '
        f'aria-label="AudBre architecture: browser and FastAPI run locally, '
        f'a Modal A10G container runs sam-audio-large in the cloud">')
    add(f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{ink}"/></marker></defs>')

    # zone labels
    add(f'<text x="30" y="34" font-size="11" font-weight="700" '
        f'letter-spacing="2.6" fill="{dim}">LOCAL</text>')
    add(f'<text x="{W - 30}" y="34" font-size="11" font-weight="700" '
        f'letter-spacing="2.6" fill="{dim}" text-anchor="end">CLOUD</text>')

    for y1, y2 in DIVIDER_SEGMENTS:
        add(f'<line x1="{DIVIDER_X}" y1="{y1}" x2="{DIVIDER_X}" y2="{y2}" '
            f'stroke="{faint}" stroke-width="1" stroke-dasharray="3 4"/>')

    # boxes
    for x, w, title, lines in BOXES:
        add(f'<rect x="{x}" y="{BOX_Y}" width="{w}" height="{BOX_H}" '
            f'fill="none" stroke="{ink}" stroke-width="1"/>')
        add(f'<rect x="{x}" y="{BOX_Y}" width="{w}" height="{HEAD_H}" fill="{ink}"/>')
        add(f'<text x="{x + 14}" y="{BOX_Y + 18}" font-size="11" font-weight="700" '
            f'letter-spacing="2.2" fill="{paper}">{esc(title)}</text>')
        for i, line in enumerate(lines):
            if not line:
                continue
            add(f'<text x="{x + 14}" y="{BOX_Y + HEAD_H + 28 + i * 22}" font-size="12" '
                f'fill="{ink if i == 0 else dim}">{esc(line)}</text>')

    # arrows
    for x1, x2, y, _, label_x, label in ARROWS:
        add(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{ink}" '
            f'stroke-width="1.2" marker-end="url(#a)"/>')
        ly = y - 13 if y == OUT_Y else y + 21
        add(f'<text x="{label_x}" y="{ly}" font-size="10" letter-spacing="0.6" '
            f'fill="{dim}" text-anchor="middle">{esc(label)}</text>')

    # caption
    add(f'<line x1="30" y1="342" x2="{W - 30}" y2="342" stroke="{faint}" stroke-width="1"/>')
    add(f'<text x="{W // 2}" y="368" font-size="12" fill="{dim}" '
        f'text-anchor="middle">{esc(CAPTION)}</text>')

    add("</svg>")
    return "\n".join(o) + "\n"


ARTBOARD = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
  <style>
    body {{ margin:0; background:{paper}; color:{ink};
           font-family:'Courier Prime','Courier New',Courier,monospace; }}
    * {{ box-sizing:border-box; }}
    a {{ color:{ink}; }} a:hover {{ color:{dim}; }}
  </style>
</helmet>
<div style="width:1200px; min-height:500px; background:{paper}; padding:44px 40px; \
display:flex; flex-direction:column;">
  <div style="display:flex; align-items:baseline; margin-bottom:6px;">
    <div style="font-size:11px; font-weight:700; letter-spacing:0.22em; color:{dim};">\
{label}</div>
    <div style="flex:1;"></div>
    <div style="font-size:10px; letter-spacing:0.12em; color:{dim};">\
RENDERED FROM design/architecture-{scheme}.svg</div>
  </div>
  <div style="height:1px; background:{faint}; margin-bottom:26px;"></div>
  {svg}
</div>
</x-dc>
</body>
</html>
"""

LABELS = {"light": "GITHUB — LIGHT THEME", "dark": "GITHUB — DARK THEME"}


def main() -> None:
    for name, palette in SCHEMES.items():
        svg = build(palette)

        path = OUT / f"architecture-{name}.svg"
        path.write_text(svg, encoding="utf-8")
        print(f"wrote {path.relative_to(OUT.parent)}  ({path.stat().st_size} bytes)")

        # The same markup on the design canvas, so a review sees what ships.
        board = OUT / f"Arch{name.capitalize()}.dc.html"
        fluid = svg.replace(f'width="{W}" height="{H}"', 'width="100%" height="auto"', 1)
        board.write_text(
            ARTBOARD.format(svg=fluid, scheme=name, label=LABELS[name], **palette),
            encoding="utf-8",
        )
        print(f"wrote {board.relative_to(OUT.parent)}  ({board.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
