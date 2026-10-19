#!/usr/bin/env python3
"""Generate animated and static light/dark README comparison SVGs.

The composite pseudocode collects independent failure patterns. Animation only
points to each correction; it does not represent one recorded repair run.
No JavaScript, external fonts, external images, or runtime service is required.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
LOOP_SECONDS = 33.0
OVERVIEW_SECONDS = 1.8
STEP_SECONDS = 4.8
FADE_SECONDS = 0.18

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


CORRECT = [
    "x = load_rows(scores,",
    "              stride=cols)",
    "x = fp32(x)",
    "s = where(valid, x, -INF)",
    "m = max(s, axis=-1)",
    "z = s - m",
    "e = exp_fp32(z)",
    "d = sum(e, axis=-1,",
    "        accumulator=FP32)",
    "p = fp16(e / d)",
]
DEFECTS = {
    2: ("index", "              stride=rows)", [("rows", "cols")]),
    4: ("mask", "s = where(valid[0], x, 0)", [("valid[0]", "valid"), (", 0)", ", -INF)")]),
    5: ("mask", "m = max(x, axis=-1)", [("x", "s")]),
    6: ("numeric", "z = s", [("s", "s - m")]),
    8: ("reduce", "d = sum(e, axis=0,", [("axis=0", "axis=-1")]),
    9: ("numeric", "        accumulator=FP16)", [("FP16", "FP32")]),
}

EXPLANATIONS = {
    1: (
        "Wrong row address",
        "stride=rows advances by the row count, so the next row loads the wrong values.",
        "stride=cols restores the correct row start before any arithmetic.",
    ),
    2: (
        "Two masking errors, one correction site",
        "A shared mask can select the wrong items; zero-filled padding still adds weight.",
        "Use each row’s mask and −INF to exclude invalid positions before normalization.",
    ),
    3: (
        "The wrong maximum breaks stability",
        "Padding’s raw 0 can outrank valid scores near −104, so valid exponentials underflow.",
        "Take max(s) after masking; the valid exponentials remain representable in this input.",
    ),
    4: (
        "Exponentials overflow or vanish",
        "Without the shift, exp(90) becomes Inf; exp(−104) rounds to zero in FP32.",
        "Subtract the row maximum to preserve relative weights and avoid these NaN-producing extremes.",
    ),
    5: (
        "Separate inputs share a denominator",
        "axis=0 mixes rows, giving each column a cross-batch denominator.",
        "Reduce axis=−1 instead: each row uses the sum of its own exponentials.",
    ),
    6: (
        "Small increments disappear",
        "Sequential FP16 rounding stalls at 2,048 when summing 4,096 ones.",
        "Accumulate in FP32 to reach 4,096; keep the final output FP16.",
    ),
}



def motion_css():
    rules = [".step-focus { opacity:0; pointer-events:none; }"]
    for step in range(1, 7):
        start = OVERVIEW_SECONDS + (step - 1) * STEP_SECONDS
        end = start + STEP_SECONDS
        marks = [(0, 0), (start, 0), (start + FADE_SECONDS, 1),
                 (end - FADE_SECONDS, 1), (end, 0), (LOOP_SECONDS, 0)]
        frames = " ".join(f"{time / LOOP_SECONDS * 100:.4f}% {{ opacity:{value}; }}" for time, value in marks)
        rules.append(f"@keyframes correction-{step} {{ {frames} }}")
        rules.append(f".step-{step} {{ animation:correction-{step} {LOOP_SECONDS:g}s linear infinite; }}")
    rules.append("@media (prefers-reduced-motion:reduce) { .step-focus { animation:none!important; opacity:0!important; } }")
    return "<style>" + "\n".join(rules) + "</style>"


def focus_overlay(p, step, y, fixed, family, tokens, *, animated, active_step, opacity):
    attr = f'class="step-focus step-{step}" opacity="0"' if animated else f'opacity="{opacity if active_step == step else 0:g}"'
    color = p[family + "_ink"]
    cy = y - 7
    parts = [f'<g {attr} data-correction="{step}" aria-hidden="true">',
        f'<rect x="34" y="{y-24}" width="547" height="32" rx="2" fill="none" stroke="{color}" stroke-width="2"/>',
        f'<rect x="700" y="{y-24}" width="547" height="32" rx="2" fill="none" stroke="{color}" stroke-width="2"/>',
        f'<path d="M595 {cy}h88m-8-6 8 6-8 6" fill="none" stroke="{color}" stroke-width="3.6"/>',
        f'<circle cx="638" cy="{cy}" r="12" fill="{color}"/>',
        f'<text x="638" y="{cy+4}" text-anchor="middle" fill="{p["paper"]}" font-size="13" font-weight="600">{step}</text>']
    for _, token in tokens:
        start = fixed.index(token)
        if token.startswith(", "):
            start += 2
            token = token[2:]
        x = 756 + start * 13.2
        parts.append(f'<path d="M{x} {y+4}h{len(token)*13.2:g}" stroke="{color}" stroke-width="2.4"/>')
    title, cause, patch = EXPLANATIONS[step]
    # The note shares the arrow's exact opacity animation and a fixed footprint.
    parts.extend([
        f'<g data-explanation="{step}">',
        f'<rect x="32" y="670" width="1216" height="100" rx="7" fill="{p[family + "_bg"]}"/>',
        f'<rect x="32" y="680" width="4" height="80" rx="2" fill="{color}"/>',
        f'<text x="52" y="696" fill="{color}" font-size="18" font-weight="600">{step:02d} / {escape(title)}</text>',
        f'<text x="52" y="725" fill="{color}" font-size="11" font-weight="600" letter-spacing=".8">CAUSE</text>',
        f'<text x="119" y="725" fill="{p["ink"]}" font-size="17">{escape(cause)}</text>',
        f'<text x="52" y="751" fill="{color}" font-size="11" font-weight="600" letter-spacing=".8">PATCH</text>',
        f'<text x="119" y="751" fill="{p["ink"]}" font-size="17">{escape(patch)}</text>',
        '</g>',
    ])
    return "\n".join(parts) + "</g>"

def make_hero(theme, *, animated=True, active_step=None, focus_opacity=1.0):
    p = PALETTES[theme]
    style = motion_css() if animated else ""
    motion_description = ("Arrows, correction sites and a cause-and-effect note below the code are highlighted together in sequence; the note is hidden between highlights and the code remains still."
                          if animated else "All code, arrows and correction sites remain visible in this static overview.")
    focus_groups = []
    neutral = "#f3f3ed" if theme == "light" else "#242c38"
    out = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="940" viewBox="0 0 1280 940" role="img" aria-labelledby="title desc">
  <title id="title">DRDebug — compare failure patterns with targeted corrections</title>
  <desc id="desc">A full ten-line Masked Softmax pseudocode block on the left collects independent defects; the right shows the corresponding correct implementation. Six arrows identify row addressing, row masks and padding, masked maximum, exponential stabilization, reduction axis and accumulator precision. Cross-tile normalization is a separate alternative below. This is a composite illustration, not one recorded repair run. {motion_description}</desc>
{style}
  <rect x="1" y="1" width="1278" height="938" rx="18" fill="{p['background']}" stroke="{p['border']}"/>
  <g font-family="Arial,Helvetica,sans-serif">
    <g fill="{p['accent']}"><rect x="36" y="37" width="9" height="9"/><rect x="49" y="37" width="9" height="9" opacity=".7"/><rect x="36" y="50" width="9" height="9" opacity=".7"/><path d="m50 55 4 4 6-8" fill="none" stroke="{p['accent']}" stroke-width="2.5"/></g>
    <text x="73" y="64" fill="{p['ink']}" font-family="Georgia,'Times New Roman',serif" font-size="47" letter-spacing="-1.5">DRDebug<tspan fill="{p['accent']}">.</tspan></text>
    <text x="378" y="52" fill="{p['accent']}" font-family="Georgia,'Times New Roman',serif" font-size="27">Don’t Regenerate, Debug.</text>
    <text x="378" y="82" fill="{p['muted']}" font-size="17">Keep the implementation. Correct the failing steps.</text>
    <text x="36" y="124" fill="{p['ink']}" font-size="23" font-weight="600">One kernel. Multiple ways to go wrong.</text>
    <text x="36" y="153" fill="{p['muted']}" font-size="16">Follow the highlighted correction and its explanation below.</text>
''']
    for x, w, label, family in [(759,103,"Indexing","index"), (872,99,"Masking","mask"), (981,111,"Reduction","reduce"), (1102,142,"Floating point","numeric")]:
        out.append(f'<rect x="{x}" y="116" width="{w}" height="29" rx="5" fill="{p[family + "_bg"]}"/><text x="{x+w/2}" y="136" text-anchor="middle" fill="{p[family + "_ink"]}" font-size="14">{label}</text>')
    for x, title, subtitle in [(32,"FAILURE PATTERNS","masked_softmax · independent defects shown together"), (698,"TARGETED CORRECTIONS","Same computation, corrected at the source")]:
        out.append(f'''<rect x="{x}" y="184" width="550" height="467" rx="9" fill="{p['paper']}" stroke="{p['border']}"/>
        <text x="{x+20}" y="213" fill="{p['ink']}" font-size="15" font-weight="600" letter-spacing="1.2">{title}</text>
        <text x="{x+20}" y="236" fill="{p['muted']}" font-size="13">{subtitle}</text>
        <path d="M{x} 250h550" stroke="{p['border']}"/>''')
    for number, fixed in enumerate(CORRECT, 1):
        y = 284 + (number - 1) * 34
        defect = DEFECTS.get(number)
        family, broken, tokens = defect if defect else (None, fixed, [])
        for x, code, corrected in [(32,broken,False), (698,fixed,True)]:
            code_x = x + 58
            bg = p[family + "_bg"] if family and not corrected else neutral if not defect else p["paper"]
            opacity = "1" if defect else ".68"
            out.append(f'<rect x="{x+1}" y="{y-25}" width="548" height="34" fill="{bg}"/>')
            if defect and not corrected:
                out.append(f'<rect x="{x+1}" y="{y-25}" width="4" height="34" fill="{p[family+"_ink"]}"/>')
            out.append(f'<text x="{x+14}" y="{y}" fill="{p["muted"]}" font-family="Consolas,\'Liberation Mono\',monospace" font-size="14">{number:02d}</text>')
            out.append(f'<text x="{code_x}" y="{y}" fill="{p["ink"]}" opacity="{opacity}" font-family="Consolas,\'Liberation Mono\',monospace" font-size="22" xml:space="preserve">{escape(code)}</text>')
            if not defect:
                continue
            sign = "+" if corrected else "−"
            sign_color = p["gain"] if corrected else p[family+"_ink"]
            out.append(f'<text x="{x+42}" y="{y}" fill="{sign_color}" font-size="17" text-anchor="middle">{sign}</text>')
            if corrected:
                for wrong_token, right_token in tokens:
                    start = code.index(right_token)
                    hx = code_x + start * 13.2
                    # Commas delimit the padding token, but are not themselves a repair.
                    if right_token.startswith(", "):
                        start += 2; right_token = right_token[2:]; hx = code_x + start * 13.2
                    width = len(right_token) * 13.2
                    out.append(f'<rect x="{hx-3}" y="{y-22}" width="{width+6}" height="28" rx="3" fill="{p[family+"_bg"]}"/>')
                    out.append(f'<text x="{hx}" y="{y}" fill="{p[family+"_ink"]}" font-family="Consolas,\'Liberation Mono\',monospace" font-size="22" font-weight="600" xml:space="preserve">{escape(right_token)}</text>')
        if defect:
            cy = y - 7
            badge = list(DEFECTS).index(number) + 1
            stroke = p[family+"_ink"]
            out.append(f'<path d="M595 {cy}h88m-8-6 8 6-8 6" fill="none" stroke="{stroke}" stroke-width="1.9"/>')
            out.append(f'<circle cx="638" cy="{cy}" r="11" fill="{p[family+"_bg"]}" stroke="{stroke}"/><text x="638" y="{cy+4}" text-anchor="middle" fill="{stroke}" font-size="12" font-weight="600">{badge}</text>')
            focus_groups.append(focus_overlay(p, badge, y, fixed, family, tokens,
                animated=animated, active_step=active_step, opacity=focus_opacity))
    out.extend(focus_groups)
    for x in [32,698]:
        out.append(f'<path d="M{x+18} 608h514" stroke="{p["border"]}"/>')
    out.append(f'''<text x="52" y="633" fill="{p['muted']}" font-size="13">Contiguous rows · ≥1 valid item/row · keepdims</text>
    <text x="718" y="633" fill="{p['muted']}" font-size="13">FP16 input/output · FP32 internal · per-row normalization</text>
    <rect x="32" y="790" width="1216" height="56" rx="7" fill="{p['reduce_bg']}"/>
    <text x="51" y="812" fill="{p['reduce_ink']}" font-size="12" font-weight="600" letter-spacing=".7">ANOTHER IMPLEMENTATION RISK · CROSS-TILE NORMALIZATION</text>
    <text x="51" y="833" fill="{p['reduce_ink']}" font-family="Consolas,'Liberation Mono',monospace" font-size="17">sum_per_tile(e)  →  sum(e, axis=-1)</text>
    <text x="1226" y="826" text-anchor="end" fill="{p['reduce_ink']}" font-size="14">Merge the denominator before normalizing.</text>
    <text x="33" y="877" fill="{p['ink']}" font-size="15">Constructed illustration: independent failure patterns shown together, not one recorded repair.</text>
    <text x="33" y="904" fill="{p['muted']}" font-size="14">Each defect is evaluated separately. Real kernels require independent NPU validation.</text>
    <text x="1246" y="904" text-anchor="end" fill="{p['accent']}" font-size="14">Explore the inputs and evidence on the website ↗</text>
  </g>
</svg>''')
    return "\n".join(out)



def export_preview_frames(destination):
    destination.mkdir(parents=True, exist_ok=True)
    loop_ms = round(LOOP_SECONDS * 1000)
    overview_ms = round(OVERVIEW_SECONDS * 1000)
    step_ms = round(STEP_SECONDS * 1000)
    fade_half_ms = round(FADE_SECONDS * 1000 / 2)
    manifest = {"loop_ms":loop_ms, "themes":{}}
    for theme in PALETTES:
        frames = []
        overview = f"{theme}-overview.svg"
        (destination / overview).write_text(make_hero(theme, animated=False), encoding="utf-8")
        frames.append({"file":overview, "duration":overview_ms, "step":None})
        for step in range(1, 7):
            half, peak = f"{theme}-step-{step}-half.svg", f"{theme}-step-{step}.svg"
            (destination / half).write_text(make_hero(theme, animated=False, active_step=step, focus_opacity=.5), encoding="utf-8")
            (destination / peak).write_text(make_hero(theme, animated=False, active_step=step), encoding="utf-8")
            # Snapshots approximate the continuous SVG fades without moving text.
            frames.extend([{"file":half,"duration":fade_half_ms,"step":step},
                           {"file":peak,"duration":step_ms - 2 * fade_half_ms,"step":step},
                           {"file":half,"duration":fade_half_ms,"step":step}])
        frames.append({"file":overview,"duration":loop_ms - overview_ms - 6 * step_ms,"step":None})
        manifest["themes"][theme] = frames
    (destination / "frames.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, help="Also export SVG snapshots and a GIF-preview timing manifest into this directory")
    args = parser.parse_args()
    target = ROOT / "assets/readme"
    for theme in PALETTES:
        for animated, suffix in [(True, ""), (False, "-static")]:
            output = target / f"hero-{theme}{suffix}.svg"
            output.write_text(make_hero(theme, animated=animated), encoding="utf-8")
            print(output.relative_to(ROOT))
    if args.preview_dir:
        export_preview_frames(args.preview_dir)


if __name__ == "__main__":
    main()
