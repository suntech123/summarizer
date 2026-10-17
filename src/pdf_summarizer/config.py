"""Configuration constants for the PDF and Markdown processing pipeline."""

from pathlib import Path
import re

CODE_GEN_PROMPT_TEMPLATE_DIR: Path = Path("data/code-gen-prompt-template")
CODE_GEN_PROMPT_INSTANCE_DIR: Path = Path("data/code-gen-prompt-instances")
MIN_EXAMPLES: int = 8
MAX_EXAMPLES: int = 12
LIBRARY_VERSIONS: str = "Python 3.11 and PyTorch 2.x"

# Layout classes whose leading '#' is a real markdown heading marker
HEADING_BOX_CLASSES: set[str] = {"title", "section-header"}

# Story scales an image down to whatever vertical space is left on the page
IMAGE_CSS: str = "img {width: 400pt;}"
IMAGE_WIDTH_PT: float = 400.0
IMAGE_LINE_RE: re.Pattern = re.compile(r"(?m)^!\[\]\([^)]*\)\s*$")

IMAGE_DPI = 300
IMAGE_FORMAT = "png"

# Pipeline directories, relative to the repository root
INPUT_DIR: Path = Path("data/original-pdf-doc")
OUTPUT_MARKDOWN_DIR: Path = Path("output-markdowns")
EXTRACTED_IMAGES_DIR: Path = Path("data/extracted-image-from-pdf-doc")
ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR: Path = Path("data/original-summary-markdown-with-image-tag")
ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_AND_CODE_TAG_DIR: Path = Path("data/original-summary-markdown-with-image-tag-and-code-tag")
SUMMARISED_INPUT_MARKDOWN_DIR: Path = Path("summarised-input-markdown")
FIGURE_ENRICHED_SUMMARY_MARKDOWN_DIR: Path = Path("figure_enriched_summary_markdown")
FINAL_OUTPUT_PDF_DIR: Path = Path("final_output_pdf")
# COHERENT_MARKDOWN_DIR: Path = Path("output-markdown-coherent")
# COHERENT_MARKDOWN_PATH: Path = COHERENT_MARKDOWN_DIR / "coherent.md"
# MERGE_PLAN_PATH: Path = COHERENT_MARKDOWN_DIR / "merge_plan.json"
IMAGE_MAP_TMP_DIR: Path = Path("data/temp/maps")
IMAGE_TAG_TO_ACTUAL_IMAGE_DIR: Path = Path("data/pdf-ready-summary-markdown-with-actual-image-and-code")
CODE_SECTION_HIERARCHY_TMP_DIR: Path = Path("data/temp/code-section-metadata")
CODE_SECTION_CONTENT_TMP_DIR: Path = Path("data/temp/code-section-metadata-with-content")
CODE_GEN_PROMPT_TEMPLATE_DIR: Path = Path("prompt/code-example-generation-template")
CODE_GEN_PROMPT_INSTANCE_DIR: Path = Path("prompt/code-example-generation-prompts")


RENDER_CSS = """
* {
    box-sizing: border-box !important;
    max-width: 100%;
}

html, body {
    margin: 0;
    padding: 0;
    font-family: "New Baskerville", "NewBaskerville", Baskerville, "Times New Roman", Times, serif;
    font-size: 10pt;
    line-height: 1.38;
    color: #111111;
    text-align: justify;
    width: 100%;
    overflow-wrap: break-word;
    word-wrap: break-word;
}

p, li, blockquote, td, th {
    font-family: inherit;
    font-size: inherit;
    line-height: inherit;
    overflow-wrap: break-word;
    word-break: break-word;
}

p {
    margin: 0 0 7pt 0;
    text-indent: 0;
}

p + p {
    text-indent: 1.25em;
}

h1, h2, h3, h4, h5, h6 {
    font-family: "Franklin Gothic Medium", "Arial Narrow", Arial, sans-serif;
    color: #0f1720;
    line-height: 1.2;
    page-break-after: avoid;
    margin-top: 12pt;
    margin-bottom: 6pt;
}

h1 {
    font-size: 19pt;
    margin-top: 0;
    margin-bottom: 10pt;
}

h2 {
    font-size: 15pt;
}

h3 {
    font-size: 12.5pt;
}

h4 {
    font-size: 11.5pt;
}

h1 + p,
h2 + p,
h3 + p,
h4 + p,
h5 + p,
h6 + p {
    text-indent: 0;
}

ul, ol {
    margin: 0 0 8pt 1.4em;
    padding-left: 1.1em;
}

li {
    margin: 2pt 0;
}

blockquote {
    margin: 8pt 0;
    padding: 2pt 0 2pt 10pt;
    border-left: 2pt solid #b8c2cc;
    color: #2f3a45;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0 10pt 0;
    page-break-inside: avoid;
    table-layout: fixed;
}

th, td {
    border: 0.5pt solid #9ea8b3;
    padding: 4pt 6pt;
    vertical-align: top;
    text-align: left;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

th {
    background-color: #eef1f5;
    font-weight: 600;
}

img {
    display: block;
    margin: 8pt auto;
    width: 400pt;
    max-width: 100% !important;
    height: auto !important;
}

pre {
    background-color: #f4f6f8;
    border-width: 0.1pt;
    border-style: solid;
    border-color: #c8d1da;
    padding: 8pt 2pt;
    margin-top: 8pt;
    margin-bottom: 8pt;
    font-family: "Courier New", Courier, monospace;
    font-size: 8.5pt;
    line-height: 1.35;
    white-space: pre-wrap;
}

pre code {
    font-family: "Courier New", Courier, monospace;
    white-space: pre-wrap;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    max-width: 100%;
    background-color: transparent;
    border: none;
    padding: 0;
}

code {
    font-family: "Courier New", Courier, monospace;
    background-color: #f9f9f9;
    padding: 1pt;
}

hr {
    border: 0;
    border-top: 0.5pt solid #c5cdd6;
    margin: 10pt 0;
}

a {
    color: #0a4b86;
    text-decoration: underline;
}
"""