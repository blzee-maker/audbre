# Design source

The AudBre interface is designed on a Claude Design canvas. These are the
sources; the published canvas is a build artifact and is not committed.

| File | Artboard |
| --- | --- |
| `Empty.dc.html` | Empty state — drop a recording |
| `Main.dc.html` | Editor — waveform, prompt, preview, removal stack |
| `Export.dc.html` | Export — before/after and the written-out report |
| `Parts.dc.html` | Type scale, controls, row and message states |
| `canvas.json` | Layout, annotations, launch view |

Page two of the canvas holds the README architecture diagram, in both GitHub
themes. Those two artboards and the shipped SVGs come from one generator, so
they cannot drift:

| File | Role |
| --- | --- |
| `make_diagram.py` | **Source of truth** — edit this, not the SVGs |
| `architecture-light.svg` | Committed, referenced by the README |
| `architecture-dark.svg` | Committed, picked by `prefers-color-scheme` |
| `ArchLight.dc.html` / `ArchDark.dc.html` | Generated artboards |

```bash
python design/make_diagram.py
```

Re-seed and publish after editing any of them:

```bash
node "<skill dir>/seed-canvas.mjs" \
  --template "<skill dir>/payload.template.html" \
  --out audbre-interface.html --title "AudBre Interface" \
  --artboard design/Main.dc.html --artboard design/Empty.dc.html \
  --artboard design/Export.dc.html --artboard design/Parts.dc.html \
  --canvas design/canvas.json
```

## Tokens

Strictly achromatic. Courier Prime throughout, with a `'Courier New', Courier,
monospace` fallback chosen for close metrics.

| Token | Value | Use |
| --- | --- | --- |
| ink | `#000000` | Text, bars, primary fills, rules |
| dim | `#767676` | Secondary text, struck-out layer names |
| faint | `#C9C9C9` | Disabled controls, dotted leaders, ghost waveform |
| rule | `#E4E4E4` | Progress track |
| hover | `#F4F4F4` | Row hover |
| paper | `#FFFFFF` | Background, inverted text |

The idea the design turns on: **removing a sound is striking a line through
it.** Kept removals are struck out and dimmed; un-check one and the text stands
back up as the sound returns to the mix.
