"""Utilities for parsing, splitting, and escaping Markdown text patterns."""

import re
from dataclasses import dataclass
from typing import Any
from config import IMAGE_LINE_RE, HEADING_BOX_CLASSES
from pathlib import Path
import json

HEADING_RE: re.Pattern = re.compile(r"^(#{1,6})\s+(.*)$")
IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\([^)]+\)$")
IMG_MARKER = re.compile(r"^\[\[img-\d+\]\]$")
TAG_LINE = re.compile(r"^[ \t]*(\[\[img-\d+\]\])[ \t]*$", re.M)

@dataclass
class Section:
    """One heading and the verbatim body text that follows it."""

    id: str
    doc: str
    level: int
    heading: str
    body: str

    def excerpt(self, limit: int = 300) -> str:
        flat = " ".join(self.body.split())
        return flat[:limit] + ("..." if len(flat) > limit else "")

    def render(self, heading: str | None = None, level: int | None = None) -> str:
        """Re-emit the section, optionally under a redesigned heading."""
        text = heading if heading is not None else self.heading
        depth = level if level is not None else self.level
        if not text:
            return self.body
        return f"{'#' * max(1, min(6, depth))} {text}\n{self.body}"

def split_into_sections(doc_index: int, doc_name: str, md_content: str) -> list[Section]:
    """Split a markdown document into heading-delimited sections.

    Text before the first heading becomes a level-0 section with no heading.
    """
    sections: list[Section] = []
    level, heading = 0, ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer)
        if heading or body.strip():
            sections.append(Section(
                id=f"D{doc_index}-S{len(sections):03d}",
                doc=doc_name,
                level=level,
                heading=heading,
                body=body,
            ))

    for line in md_content.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            buffer.append(line)
            continue
        flush()
        level, heading = len(match.group(1)), match.group(2).strip()
        buffer = []
    flush()

    return sections


def split_at_images(md_content: str) -> list[str]:
    """Split markdown so every part after the first begins with an image line."""
    bounds = [0] + [m.start() for m in IMAGE_LINE_RE.finditer(md_content)] + [len(md_content)]
    return [md_content[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def escape_literal_hashes(pages: list[dict[str, Any]]) -> str:
    """Escape line-leading '#' outside heading boxes.

    Ensures literal hashes in body/code text (e.g. "# In[1]:") are not
    rendered as structural markdown headings.
    """
    parts: list[str] = []

    for page in pages:
        text: str = page.get("text", "")
        cursor = 0
        chunks: list[str] = []
        boxes = sorted(page.get("page_boxes") or [], key=lambda b: b["pos"][0])

        for box in boxes:
            start, stop = box["pos"]
            if start > cursor:
                chunks.append(text[cursor:start])

            segment = text[start:stop]
            if box.get("class") in HEADING_BOX_CLASSES:
                # Keep the generated ATX prefix, escape literal '#' following it
                segment = re.sub(r"(?m)^(#{1,6} +)#", r"\1\\#", segment)
            else:
                segment = re.sub(r"(?m)^(\s*)#", r"\1\\#", segment)

            chunks.append(segment)
            cursor = stop

        chunks.append(text[cursor:])
        parts.append("".join(chunks))

    return "".join(parts)


### To map images
def extract_figure_image_mapping(md_content: str) -> dict[str, str]:
    """Extracts a dictionary mapping figure keys to image markdown lines."""
    lines = md_content.splitlines()
    result = {}

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        # Step 1: Key is line starting with "Fig " or "Figure "
        if stripped_line.startswith("Fig ") or stripped_line.startswith("Figure "):
            #key = line
            key = stripped_line  # Use stripped line as key
            val = None

            # Step 2 & 3 & 4: Search above first
            above_idx = i - 1
            while above_idx >= 0:
                prev_line = lines[above_idx].strip()
                if prev_line:  # Nearest non-empty line
                    if prev_line.startswith("![](") and prev_line.endswith(")"):
                        val = lines[above_idx]
                    break  # Stop at first non-empty line
                above_idx -= 1

            # Step 4: Search below ONLY if not found above
            if val is None:
                below_idx = i + 1
                while below_idx < len(lines):
                    next_line = lines[below_idx].strip()
                    if next_line:  # Nearest non-empty line
                        if next_line.startswith("![](") and next_line.endswith(")"):
                            val = lines[below_idx]
                        break  # Stop at first non-empty line
                    below_idx += 1

            if val is not None:
                result[key] = val

    return result

### Figure enrichment utility to summarised markdown
def enrich_markdown_with_figures(input_path: Path, output_path: Path, figure_mapping: dict[str, str]) -> None:
    """Reads a markdown file, inserts missing figure images before keys, and writes to output."""
    if not input_path.exists():
        return

    content = input_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    enriched_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # Identify the Key pattern
        if stripped_line.startswith("Fig ") or stripped_line.startswith("Figure "):
            # If the trimmed key exists in our mapping, insert the Value (image) before the Key
            if stripped_line in figure_mapping:
                enriched_lines.append(figure_mapping[stripped_line])
                
        # Append the original line (the Key or any standard text) after
        enriched_lines.append(line)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(enriched_lines), encoding="utf-8")


#### Extract image lines from markdown files in a folder
def image_lines(folder):
    """Yield every standalone ![](...) line from the .md files in folder, in order of appearance."""
    for md in sorted(Path(folder).glob("*.md")):
        for line in md.read_text(encoding="utf-8").splitlines():
            if IMAGE_LINE.match(line.strip()):
                yield line.strip()

#### Extract image markers from markdown files in a folder
def img_markers(folder):
    """Yield every standalone [[img-NN]] line from the .md files in folder, in order of appearance."""
    for md in sorted(Path(folder).glob("*.md")):
        for line in md.read_text(encoding="utf-8").splitlines():
            if IMG_MARKER.match(line.strip()):
                yield line.strip()


def build_img_map(llm_img_lst, original_img_lst):
    """Map each [[img-NN]] marker to the original image line at the same position."""
    if len(llm_img_lst) != len(original_img_lst):
        raise ValueError(f"length mismatch: {len(llm_img_lst)} markers vs {len(original_img_lst)} images")
    return dict(zip(llm_img_lst, original_img_lst))

def save_dict(d, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)   # swap in one step, so a crash never leaves a half-written file

def load_dict(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replace_img_tags(md_dir, out_dir, map_dir):
    """Replace standalone [[img-NN]] lines with the original image lines from the JSON image map."""
    md_dir, out_dir, map_dir = Path(md_dir), Path(out_dir), Path(map_dir)
    if out_dir.resolve() == md_dir.resolve():
        raise ValueError("out_dir must differ from md_dir (it would overwrite the inputs)")
    out_dir.mkdir(parents=True, exist_ok=True)
    for md in sorted(md_dir.glob("*.md")):
        map_file = next((p for p in (map_dir / f"{md.stem}.img_map.json", map_dir / "img_map.json")
                         if p.exists()), None)
        if map_file is None:
            print(f"skipped {md.name}: no image map found in {map_dir}")
            continue
        img_map = json.loads(map_file.read_text(encoding="utf-8"))
        missing = []

        def sub(m):
            tag = m.group(1)
            if tag in img_map:
                return img_map[tag] + "\n"      # extra newline keeps the caption in its own paragraph
            missing.append(tag)
            return m.group(0)

        text = TAG_LINE.sub(sub, md.read_text(encoding="utf-8"))
        (out_dir / md.name).write_text(text, encoding="utf-8")
        print(f"{md.name}: replaced {text.count('![](')} image(s)"
              + (f", unmapped tags: {missing}" if missing else ""))