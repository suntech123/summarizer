#!/usr/bin/env python3
"""Render markdown content or markdown files to PDF.

This module keeps markdown_pdf-specific rendering logic separate from
PDF-to-markdown extraction flows.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pymupdf
from config import FIGURE_ENRICHED_SUMMARY_MARKDOWN_DIR, FINAL_OUTPUT_PDF_DIR, RENDER_CSS
from markdown_pdf import MarkdownPdf, Section

PT_PER_MM = 72.0 / 25.4
PAPER_SIZE = "A4"

# Print-style fixed margins for digital book output.
MARGIN_TOP_MM = 20.0
MARGIN_BOTTOM_MM = 24.0
MARGIN_LEFT_MM = 25.0
MARGIN_RIGHT_MM = 10.0

# Convert margins from millimeters to points
MARGIN_TOP_PT = MARGIN_TOP_MM * PT_PER_MM
MARGIN_BOTTOM_PT = MARGIN_BOTTOM_MM * PT_PER_MM
MARGIN_LEFT_PT = MARGIN_LEFT_MM * PT_PER_MM
MARGIN_RIGHT_PT = MARGIN_RIGHT_MM * PT_PER_MM

# markdown_pdf API signature expects: (left, top, right, bottom)
SECTION_BORDERS = (
    MARGIN_LEFT_PT,
    MARGIN_TOP_PT,
    -MARGIN_RIGHT_PT,
    -MARGIN_BOTTOM_PT,
)

# markdown_pdf adds these to the page rect, so right/bottom must be negative
SECTION_BORDERS = (
    MARGIN_LEFT_PT,
    MARGIN_TOP_PT,
    -MARGIN_RIGHT_PT,
    -MARGIN_BOTTOM_PT,
)

IMAGE_WIDTH_PT = 400.0
IMAGE_LINE_RE = re.compile(r"(?m)^!\[[^\]]*\]\(([^)]*)\)\s*$")


def _split_at_images(markdown_content: str) -> list[str]:
    """Split markdown so every part after the first begins with an image line."""
    bounds = [0] + [m.start() for m in IMAGE_LINE_RE.finditer(markdown_content)] + [len(markdown_content)]
    return [markdown_content[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]

import textwrap
'''
def _wrap_markdown_code_blocks(markdown_content: str, max_chars: int = 70) -> str:
    """Hard-wrap text inside markdown code blocks to prevent PDF right-margin truncation."""
    lines = markdown_content.splitlines()
    in_code_block = False
    out_lines = []
    
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            out_lines.append(line)
            continue
            
        if in_code_block:
            # Wrap long lines inside code blocks. break_long_words=True ensures 
            # continuous strings like hashes or paths are forcefully split.
            if len(line) > max_chars:
                wrapped = textwrap.fill(line, width=max_chars, break_long_words=True, break_on_hyphens=True)
                out_lines.extend(wrapped.splitlines())
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)
            
    return "\n".join(out_lines)
'''

def _wrap_markdown_code_blocks(markdown_content: str, max_chars: int = 85) -> str:
    """Hard-wrap long lines inside fenced code blocks, preserving indentation."""
    out, in_code = [], False
    for line in markdown_content.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
        elif in_code and len(line) > max_chars:
            indent = len(line) - len(line.lstrip(" "))
            out.append(line[:max_chars])
            rest = line[max_chars:]
            step = max_chars - indent - 2
            while rest:
                out.append(" " * (indent + 2) + rest[:step])
                rest = rest[step:]
        else:
            out.append(line)
    return "\n".join(out)


def render_markdown_to_pdf(markdown_content: str, out_path: str | Path, max_passes: int = 6) -> None:
    """Render markdown to PDF, inserting breaks before squeezed images."""
    # Inject the hard-wrap processor here
    #markdown_content = _wrap_markdown_code_blocks(markdown_content, max_chars=70)
    markdown_content = _wrap_markdown_code_blocks(markdown_content, max_chars=85)
    #markdown_content = _wrap_markdown_code_blocks(markdown_content, max_chars=70)
    parts = _split_at_images(markdown_content)
    break_before: set[int] = set()
    output_path = Path(out_path)

    for _ in range(max_passes):
        chunks, current = [], parts[0]
        for i, part in enumerate(parts[1:]):
            if i in break_before:
                chunks.append(current)
                current = part
            else:
                current += part
        chunks.append(current)

        pdf = MarkdownPdf()
        pdf.meta["title"] = output_path.stem
        pdf.meta["creator"] = "markdown_to_pdf.py"
        for chunk in chunks:
            pdf.add_section(
                Section(
                    chunk,
                    toc=False,
                    root=".",
                    paper_size=PAPER_SIZE,
                    borders=SECTION_BORDERS,
                ),
                user_css=RENDER_CSS,
            )
        pdf.save(output_path)

        squeezed: set[int] = set()
        index = 0
        with pymupdf.open(output_path) as doc:
            for page in doc:
                for img in page.get_images(full=True):
                    if page.get_image_bbox(img).width < IMAGE_WIDTH_PT - 2:
                        squeezed.add(index)
                    index += 1

        if not squeezed:
            break
        break_before |= squeezed


def render_markdown_file(input_path: Path, output_path: Path, max_passes: int = 6) -> None:
    """Read markdown from disk and render it to a PDF file."""
    markdown_content = input_path.read_text(encoding="utf-8")
    render_markdown_to_pdf(markdown_content, output_path, max_passes=max_passes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render markdown to PDF using markdown_pdf with anti-squeeze image "
            "handling. Provide <input> and <output> for single-file mode, or "
            "omit both to process all files from input folder into output folder."
        )
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input markdown file")
    parser.add_argument("output", nargs="?", type=Path, help="Output PDF file")
    parser.add_argument(
        "--max-passes",
        type=int,
        default=6,
        help="Maximum render passes used to avoid squeezed images (default: 6)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=FIGURE_ENRICHED_SUMMARY_MARKDOWN_DIR,
        help="Directory containing markdown files for batch rendering",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FINAL_OUTPUT_PDF_DIR,
        help="Directory where PDF outputs are written",
    )

    args = parser.parse_args()
    if args.max_passes < 1:
        parser.error("--max-passes must be >= 1")

    single_file_mode = args.input is not None or args.output is not None
    if single_file_mode and (args.input is None or args.output is None):
        parser.error("Provide both <input> and <output> for single-file mode.")

    if args.input is not None and args.output is not None:
        if not args.input.is_file():
            parser.error(f"Input markdown file does not exist: {args.input}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        render_markdown_file(args.input, args.output, max_passes=args.max_passes)
        print(f"Created: {args.output}")
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_files = [f for f in input_dir.glob("*.md") if f.is_file()]
    if not markdown_files:
        parser.error(f"No markdown files found in input directory: {input_dir}")

    for markdown_file in markdown_files:
        output_path = output_dir / f"{markdown_file.stem}.pdf"
        render_markdown_file(markdown_file, output_path, max_passes=args.max_passes)
        print(f"Created: {output_path}")

    print(f"Processed markdown files: {len(markdown_files)}")


if __name__ == "__main__":
    main()