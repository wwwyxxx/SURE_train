#!/usr/bin/env python3
"""Convert manual.md to a PDF with Chinese font support."""
import os
import sys
from pathlib import Path

import markdown
from weasyprint import HTML, CSS


BASE_DIR = Path(__file__).resolve().parent
MANUAL_MD = BASE_DIR / "manual.md"
MANUAL_PDF = BASE_DIR / "swift-adapter-agent-manual.pdf"
FONT_REGULAR = BASE_DIR / ".pdf-fonts" / "NotoSansSC-Regular.ttf"
FONT_BOLD = BASE_DIR / ".pdf-fonts" / "NotoSansSC-Bold.ttf"


def build_css(regular_url: str, bold_url: str) -> str:
    return f"""
@page {{
    size: A4;
    margin: 2cm 1.8cm;
    @bottom-center {{
        content: counter(page);
        font-family: 'Noto Sans SC', sans-serif;
        font-size: 9pt;
        color: #666;
    }}
}}

@font-face {{
    font-family: 'Noto Sans SC';
    src: url('{regular_url}');
    font-weight: normal;
    font-style: normal;
}}

@font-face {{
    font-family: 'Noto Sans SC';
    src: url('{bold_url}');
    font-weight: bold;
    font-style: normal;
}}

body {{
    font-family: 'Noto Sans SC', sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #222;
}}

h1, h2, h3, h4 {{
    font-family: 'Noto Sans SC', sans-serif;
    color: #1a1a1a;
    margin-top: 1.4em;
    margin-bottom: 0.6em;
    page-break-after: avoid;
}}

h1 {{ font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 0.2em; }}
h2 {{ font-size: 16pt; border-bottom: 1px solid #ccc; padding-bottom: 0.15em; }}
h3 {{ font-size: 13pt; color: #333; }}
h4 {{ font-size: 11pt; color: #444; }}

p {{ margin: 0.6em 0; }}

a {{ color: #0366d6; text-decoration: none; }}

code {{
    font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace;
    font-size: 9.5pt;
    background: #f5f5f5;
    padding: 0.1em 0.3em;
    border-radius: 3px;
}}

pre {{
    background: #f8f8f8;
    border: 1px solid #e1e1e8;
    border-radius: 4px;
    padding: 0.8em;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-size: 9pt;
    line-height: 1.5;
    page-break-inside: avoid;
}}

pre code {{
    background: transparent;
    padding: 0;
    font-size: 9pt;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
    page-break-inside: avoid;
}}

th, td {{
    border: 1px solid #d0d0d0;
    padding: 0.45em 0.6em;
    text-align: left;
    vertical-align: top;
}}

th {{
    background: #f0f0f0;
    font-weight: bold;
}}

tr:nth-child(even) {{ background: #fafafa; }}

ul, ol {{
    margin: 0.5em 0;
    padding-left: 1.8em;
}}

li {{ margin: 0.25em 0; }}

blockquote {{
    border-left: 4px solid #ddd;
    padding-left: 1em;
    color: #555;
    margin: 1em 0;
}}

hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 1.5em 0;
}}
"""


def main():
    if not MANUAL_MD.exists():
        print(f"[ERROR] {MANUAL_MD} not found", file=sys.stderr)
        sys.exit(1)
    if not FONT_REGULAR.exists() or not FONT_BOLD.exists():
        print(f"[ERROR] Font files not found", file=sys.stderr)
        sys.exit(1)

    md_text = MANUAL_MD.read_text(encoding="utf-8")

    html_body = markdown.markdown(
        md_text,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
        ],
    )

    regular_url = FONT_REGULAR.as_uri()
    bold_url = FONT_BOLD.as_uri()
    css_text = build_css(regular_url, bold_url)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ms-swift 模型适配 Agent Harness 使用手册</title>
<style>
{css_text}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    HTML(string=html, base_url=str(BASE_DIR)).write_pdf(str(MANUAL_PDF))

    print(f"[OK] PDF generated: {MANUAL_PDF}")
    size = MANUAL_PDF.stat().st_size
    print(f"[INFO] File size: {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
