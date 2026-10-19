#!/usr/bin/env python3
"""Generate the self-contained light/dark README hero SVGs.

The numerical story is the website's ``sum_rounding`` fixture: one row of
4,096 valid zero logits, sequential FP16 versus FP32 accumulation, and FP16
outputs. This is a constructed CPU illustration, not a measured NPU result.
"""

from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
PALETTES = {
    "light": {
        "background": "#f7f4ee", "border": "#ced4df", "ink": "#232b38",
        "muted": "#59677c", "accent": "#6a50a8", "paper": "#fffdfa",
        "shadow": "#e5e0d9", "loss": "#9b503d", "loss_bg": "#f6e8e1",
        "gain": "#416c51", "gain_bg": "#e9f0e7", "pill": "#eee9f6",
        "index_bg": "#e4eefb", "index_ink": "#3e628a",
        "mask_bg": "#f7f0dd", "mask_ink": "#7b6532",
        "reduce_bg": "#f0eaf7", "reduce_ink": "#695184",
        "numeric_bg": "#f9e3dc", "numeric_ink": "#8b513b",
    },
    "dark": {
        "background": "#171c25", "border": "#3d485b", "ink": "#e9ebef",
        "muted": "#b0bbcb", "accent": "#b5a1f0", "paper": "#212936",
        "shadow": "#10151c", "loss": "#edb19b", "loss_bg": "#3c2e2c",
        "gain": "#afd3b7", "gain_bg": "#293d33", "pill": "#352e49",
        "index_bg": "#26364b", "index_ink": "#a6c3e8",
        "mask_bg": "#3e392b", "mask_ink": "#dfcc96",
        "reduce_bg": "#352e49", "reduce_ink": "#c8b3e4",
        "numeric_bg": "#402e2b", "numeric_ink": "#edb59f",
    },
}


def make_hero(p):
    chunks = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="580" viewBox="0 0 1200 580" role="img" aria-labelledby="title desc">
  <title id="title">DRDebug — Don’t Regenerate, Debug</title>
  <desc id="desc">Evidence-guided repair of near-miss accelerator kernels. In a constructed Masked Softmax CPU example with 4,096 valid zero scores, sequential FP16 accumulation stops at 2,048: output weights total 200 percent. Changing only the accumulator to FP32 yields a denominator of 4,096 and weights totaling 100 percent. Inputs and outputs remain FP16. Real kernels require independent NPU validation.</desc>
  <rect x="1" y="1" width="1198" height="578" rx="18" fill="{p['background']}" stroke="{p['border']}"/>
  <g fill="{p['accent']}"><rect x="48" y="49" width="10" height="10"/><rect x="63" y="49" width="10" height="10" opacity=".72"/><rect x="48" y="64" width="10" height="10" opacity=".72"/><path d="m64 69 4 4 7-10" stroke="{p['accent']}" stroke-width="3" fill="none"/></g>
  <g font-family="Arial, Helvetica, sans-serif">
    <text x="91" y="67" fill="{p['muted']}" font-size="14" letter-spacing="1.6">ACCELERATOR KERNEL REPAIR</text>
    <text x="48" y="183" fill="{p['ink']}" font-family="Georgia, 'Times New Roman', serif" font-size="86" letter-spacing="-2">DRDebug<tspan fill="{p['accent']}">.</tspan></text>
    <text x="51" y="239" fill="{p['accent']}" font-family="Georgia, 'Times New Roman', serif" font-size="34">Don’t Regenerate, Debug.</text>
    <text x="51" y="299" fill="{p['ink']}" font-size="22">Repair near-miss accelerator kernels.</text>
    <text x="51" y="335" fill="{p['muted']}" font-size="22">Evidence guides the edit.</text>
    <text x="51" y="368" fill="{p['muted']}" font-size="22">Validation decides what stays.</text>
    <path d="M51 406H510" stroke="{p['border']}"/>
''']
    for x, width, label, family in (
        (51, 99, "Indexing", "index"), (159, 99, "Masking", "mask"),
        (267, 112, "Reduction", "reduce"), (388, 122, "Numerics", "numeric"),
    ):
        chunks.append(f'''    <rect x="{x}" y="428" width="{width}" height="32" rx="5" fill="{p[family + '_bg']}"/>
    <text x="{x + width / 2:g}" y="449" text-anchor="middle" fill="{p[family + '_ink']}" font-size="14">{escape(label)}</text>
''')
    chunks.append(f'''    <text x="51" y="503" fill="{p['muted']}" font-size="13" letter-spacing="1.4">REPRODUCE · DIAGNOSE · PATCH · VERIFY</text>
    <text x="51" y="532" fill="{p['muted']}" font-size="15">Keep the kernel. Repair the cause.</text>
  </g>
  <rect x="559" y="44" width="594" height="507" rx="10" fill="{p['shadow']}"/>
  <rect x="551" y="36" width="594" height="507" rx="10" fill="{p['paper']}" stroke="{p['border']}"/>
  <g font-family="Arial, Helvetica, sans-serif">
    <text x="578" y="68" fill="{p['muted']}" font-size="13" letter-spacing="1.2">REPAIR NOTE / MASKED SOFTMAX</text>
    <path d="M578 86H1118" stroke="{p['border']}"/>
    <text x="578" y="121" fill="{p['ink']}" font-family="Georgia, 'Times New Roman', serif" font-size="27">Why can weights total 200%?</text>
    <text x="578" y="155" fill="{p['muted']}" font-size="18">4,096 valid positions · all scores are zero</text>
    <text x="578" y="188" fill="{p['ink']}" font-family="'Courier New', monospace" font-size="22">score = 0 → exp(score) = 1</text>
    <rect x="578" y="207" width="540" height="37" rx="3" fill="{p['loss_bg']}"/>
    <rect x="578" y="249" width="540" height="37" rx="3" fill="{p['gain_bg']}"/>
    <text x="592" y="232" fill="{p['loss']}" font-family="'Courier New', monospace" font-size="22">− d = sum(e, accumulator=FP16)</text>
    <text x="592" y="274" fill="{p['gain']}" font-family="'Courier New', monospace" font-size="22">+ d = sum(e, accumulator=FP32)</text>
    <text x="578" y="319" fill="{p['muted']}" font-size="13" letter-spacing="1">OUTPUT WEIGHT TOTAL · EXPECTED 100%</text>
    <path d="M845 335V415" stroke="{p['border']}"/>
    <text x="578" y="344" fill="{p['loss']}" font-size="13" letter-spacing="1">BEFORE</text>
    <text x="578" y="389" fill="{p['loss']}" font-family="Georgia, 'Times New Roman', serif" font-size="49">200%</text>
    <text x="578" y="415" fill="{p['muted']}" font-family="'Courier New', monospace" font-size="18">denominator: 2,048</text>
    <path d="M775 372h39m-8-6 8 6-8 6" fill="none" stroke="{p['accent']}" stroke-width="2.5"/>
    <text x="872" y="344" fill="{p['gain']}" font-size="13" letter-spacing="1">AFTER</text>
    <text x="872" y="389" fill="{p['gain']}" font-family="Georgia, 'Times New Roman', serif" font-size="49">100%</text>
    <text x="872" y="415" fill="{p['muted']}" font-family="'Courier New', monospace" font-size="18">denominator: 4,096</text>
    <path d="M578 435H1118" stroke="{p['border']}"/>
    <text x="578" y="461" fill="{p['ink']}" font-size="17">Sequential FP16 rounding: 2,048 + 1 → 2,048.</text>
    <text x="578" y="490" fill="{p['muted']}" font-size="15">FP16 input / output. Only the accumulator changes.</text>
    <text x="578" y="517" fill="{p['muted']}" font-size="14">Constructed CPU example · NPU validation required.</text>
  </g>
</svg>
''')
    return "".join(chunks)


def main():
    target = ROOT / "assets" / "readme"
    for name, palette in PALETTES.items():
        output = target / f"hero-{name}.svg"
        output.write_text(make_hero(palette), encoding="utf-8")
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
