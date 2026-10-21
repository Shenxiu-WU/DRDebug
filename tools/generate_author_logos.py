#!/usr/bin/env python3
"""Frame original logos as compact, visibly linked README cards.

Run from any directory. This adds a white presentation surface and whitespace;
the source logos retain their proportions, colours and original file bytes.
"""

from base64 import b64encode
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "readme" / "affiliations"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def image(filename, media_type, x, y, width, height):
    payload = b64encode((ASSETS / "sources" / filename).read_bytes()).decode("ascii")
    return (
        f'<image x="{x}" y="{y}" width="{width}" height="{height}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'xlink:href="data:{media_type};base64,{payload}"/>'
    )


def cityu():
    logo = ET.parse(ASSETS / "sources" / "cityuhk.svg").getroot()
    for key, value in {"x": "90", "y": "7", "width": "180", "height": "101"}.items():
        logo.set(key, value)
    return ET.tostring(logo, encoding="unicode")


def card(slug, title, label, artwork):
    source = f'''<svg xmlns="{SVG}" xmlns:xlink="http://www.w3.org/1999/xlink" width="360" height="172" viewBox="0 0 360 172" role="img" aria-labelledby="title desc">
  <title id="title">{escape(title)}</title>
  <desc id="desc">Original logo on a white surface for readability in light and dark README themes.</desc>
  <rect x="1" y="1" width="358" height="170" rx="12" fill="#ffffff" stroke="#d0d7de" stroke-width="2"/>
  {artwork}
  <line x1="24" y1="118" x2="336" y2="118" stroke="#eaeef2"/>
  <text x="180" y="150" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="24" fill="#0969da" text-decoration="underline">{escape(label)} ↗</text>
</svg>
'''
    (ASSETS / f"{slug}.svg").write_text(source, encoding="utf-8")


def main():
    card("cityuhk", "City University of Hong Kong", "CityUHK", cityu())
    card("optima-group", "Optima Group — Learning and Optimization for Machine Intelligence",
         "Optima Group", image("optima-group.jpg", "image/jpeg", 47, 29, 266, 54))
    card("huawei", "Huawei — CANN community", "Huawei · CANN",
         image("huawei.png", "image/png", 72, 32, 216, 49))
    print("Generated 3 README logo cards; original source files are unchanged.")


if __name__ == "__main__":
    main()
