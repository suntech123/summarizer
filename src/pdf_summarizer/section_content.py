"""Attach exact markdown text to a JSON section hierarchy.

Given a JSON file describing a section hierarchy (titles only) and a markdown
file, add ``header_level`` and ``content`` keys to every node, at any depth.
``content`` is the verbatim markdown between the node's heading line (not
included, the title already holds it) and the next heading of ANY kind, so a
parent holds only its own text and headings missing from the JSON are dropped.
``header_level`` is the number of ``#`` characters before the heading line.

A second file is written too: the same markdown in which each matched section
keeps its heading line and has its body replaced by one tag line,
``[[content - Section/Subsection]]`` (titles joined by "/"), e.g.::

    ## 4.1 Working with images
    [[content - 4.1 Working with images]]
    ### 4.1.1 Adding color channels
    [[content - 4.1 Working with images/4.1.1 Adding color channels]]

Replacing a tag line with that node's ``content`` restores the section.

Folders come from the ``config`` module (same directory as this file); each one
must be a ``pathlib.Path``:
    CODE_SECTION_HIERARCHY_TMP_DIR                       -> hierarchy JSON is read from here
    ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR         -> markdown is read from here
    CODE_SECTION_CONTENT_TMP_DIR                         -> output JSON is written here
    ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_AND_CODE_TAG_DIR -> tagged markdown is written here

Usage:
    python section_content.py hierarchy.json document.md [-o out.json] [-t tagged.md]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
from bisect import bisect_left
from pathlib import Path
from typing import Any, Optional, Union

from config import (
    CODE_SECTION_CONTENT_TMP_DIR,
    CODE_SECTION_HIERARCHY_TMP_DIR,
    ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_AND_CODE_TAG_DIR,
    ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR,
)

log = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]

# --------------------------------------------------------------------------- #
# Markdown heading scanner (CommonMark-aware, single pass, O(n))
# --------------------------------------------------------------------------- #
# One match per physical line; handles \r\n, \n and lone \r so offsets stay exact.
_LINE = re.compile(r"([^\r\n]*)(\r\n|\r|\n)?")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")
_ATX = re.compile(r" {0,3}(#{1,6})(?=[ \t]|$)(.*)")
_ATX_CLOSE = re.compile(r"(?:^|[ \t]+)#+[ \t]*$")
_SETEXT = re.compile(r" {0,3}(?:=+|-+)[ \t]*")
_BREAK = re.compile(r" {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*")
_BLOCK = re.compile(r" {0,3}(?:>|[-+*](?:[ \t]|$)|\d{1,9}[.)](?:[ \t]|$))")

# Light inline-markup clean-up so "## **3.2** `Foo`" still matches "3.2 Foo".
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")

# Blank lines (only) at the start of a section body; never eats first-line indentation.
_LEADING_BLANK = re.compile(r"(?:[ \t]*(?:\r\n|\r|\n))*")


def _normalize(title: str) -> str:
    title = unicodedata.normalize("NFKC", title)  # also turns NBSP into a space
    title = _LINK.sub(r"\1", title)
    title = _ESCAPE.sub(r"\1", title)
    title = title.replace("`", "").replace("*", "")
    return " ".join(title.split()).casefold()


def _scan_headings(text: str) -> list[tuple[str, int, int, int]]:
    """Return ``[(normalized_title, start, level, body_start), ...]`` in document order.

    ``start`` is the offset of the heading's first line, ``body_start`` the offset
    just after its last line (and line break).  ``level`` is the number of ``#``
    for ATX headings; setext headings have no ``#`` so they report their
    equivalent level (1 for ``===``, 2 for ``---``).

    Ignores anything inside fenced code blocks (so ``# In [1]:`` is not a
    heading), supports ATX (``## x``, optional closing ``#``s) and setext
    (underlined, including multi-line) headings, and never treats 4+ space
    indented lines as headings.
    """
    headings: list[tuple[str, int, int, int]] = []
    fence: Optional[tuple[str, int]] = None  # (fence char, opening length)
    para: list[tuple[int, str]] = []  # open paragraph: (line_start, stripped_text)

    for m in _LINE.finditer(text):
        start = m.start()
        if start == m.end():  # empty match at EOF
            continue
        line = m.group(1)

        if fence:  # inside a fenced block: only look for the closing fence
            fm = _FENCE.fullmatch(line)
            if (
                fm
                and fm.group(1)[0] == fence[0]
                and len(fm.group(1)) >= fence[1]
                and not fm.group(2).strip()
            ):
                fence = None
            continue

        if not line.strip():
            para.clear()
            continue

        if para and _SETEXT.fullmatch(line):
            title = " ".join(t for _, t in para)
            level = 1 if line.lstrip().startswith("=") else 2
            headings.append((_normalize(title), para[0][0], level, m.end()))
            para.clear()
            continue

        am = _ATX.fullmatch(line)
        if am:
            title = _ATX_CLOSE.sub("", am.group(2)).strip()
            headings.append((_normalize(title), start, len(am.group(1)), m.end()))
            para.clear()
            continue

        fm = _FENCE.fullmatch(line)
        if fm and not (fm.group(1)[0] == "`" and "`" in fm.group(2)):
            fence = (fm.group(1)[0], len(fm.group(1)))
            para.clear()
            continue

        if _BREAK.fullmatch(line) or _BLOCK.match(line):
            para.clear()
            continue

        if not para and (line.startswith("    ") or line.startswith("\t")):
            continue  # indented code block, not a paragraph

        para.append((start, line.strip()))

    return headings


# --------------------------------------------------------------------------- #
# Hierarchy handling
# --------------------------------------------------------------------------- #
def _build_skeleton(
    roots: list,
    title_key: str,
    children_key: str,
    content_key: str,
    level_key: str,
    root_label: str,
) -> tuple[list, list[tuple[dict, str, str]]]:
    """Copy the hierarchy and return (new_roots, nodes_in_preorder).

    Each entry of ``nodes_in_preorder`` is ``(new_node, title, tag_path)``, where
    ``tag_path`` is the titles from the top-level section down to the node,
    joined by "/".

    Iterative (no recursion limit) and non-mutating.  Each new node is
    ``{title, header_level: None, content: None, <other keys...>, subsections: [...]}``
    with the original key order preserved and the two new keys placed right
    after the title.
    """
    new_roots: list = []
    flat: list[tuple[dict, str, str]] = []
    stack = [(iter(enumerate(roots)), new_roots, root_label, "")]

    while stack:
        it, dst, label, prefix = stack[-1]
        for i, node in it:
            here = f"{label}[{i}]"
            if not isinstance(node, dict):
                raise ValueError(f"{here}: expected an object, got {type(node).__name__}")
            title = node.get(title_key)
            if not isinstance(title, str):
                raise ValueError(f"{here}: missing or non-string {title_key!r}")

            out: dict[str, Any] = {}
            kids_out: Optional[list] = None
            for k, v in node.items():
                if k == content_key or k == level_key:
                    continue  # replaced below
                if k == children_key:
                    if not isinstance(v, list):
                        raise ValueError(f"{here}.{children_key}: expected a list")
                    kids_out = out[k] = []
                else:
                    out[k] = v
                if k == title_key:
                    out[level_key] = None
                    out[content_key] = None
            clean = " ".join(title.split())  # tag-safe: no newlines / space runs
            path = f"{prefix}/{clean}" if prefix else clean
            dst.append(out)
            flat.append((out, title, path))

            if kids_out is not None and node[children_key]:
                stack.append(
                    (
                        iter(enumerate(node[children_key])),
                        kids_out,
                        f"{here}.{children_key}",
                        path,
                    )
                )
                break  # descend: keeps pre-order == document order
        else:
            stack.pop()

    return new_roots, flat


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, data: str) -> None:
    """Write text via temp file + rename; ``newline=""`` leaves CRLF/LF untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def attach_section_content(
    hierarchy_file: PathLike,
    markdown_file: PathLike,
    output_file: Optional[PathLike] = None,
    tagged_markdown_file: Optional[PathLike] = None,
    *,
    title_key: str = "title",
    children_key: str = "subsections",
    content_key: str = "content",
    level_key: str = "header_level",
    root_key: str = "sections",
    strict: bool = False,
    trim: bool = True,
    encoding: str = "utf-8",
) -> dict:
    """Add the header level and exact markdown text of every section to a JSON hierarchy.

    Args:
        hierarchy_file: name of the JSON file inside CODE_SECTION_HIERARCHY_TMP_DIR:
            ``{"sections": [...]}`` or a bare list, where each node has a title
            and optional nested ``subsections`` (any depth).
        markdown_file: name of the markdown file inside
            ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR.
        output_file: name of the result inside CODE_SECTION_CONTENT_TMP_DIR
            (created if missing).  Defaults to the hierarchy file's name.
            An absolute path given for any of these files is used as-is.
        tagged_markdown_file: name of the second output, inside
            ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_AND_CODE_TAG_DIR (created if
            missing): the markdown with each matched section's body replaced by
            ``[[content - Section/Subsection]]`` on the line right after the
            (kept) heading.  Sections that are not in the JSON are left
            untouched.  Defaults to the markdown file's name.
        title_key / children_key / content_key / level_key / root_key: JSON key names.
        strict: raise ``ValueError`` if any title is not found in the markdown;
            otherwise its ``content`` and ``header_level`` are ``None`` and a
            warning is logged.
        trim: drop the blank lines between the heading and the body and any
            trailing whitespace at the end of the body (a heading with no body
            gives ``""``).  ``False`` keeps the byte-exact text that follows the
            heading line, up to the next heading.
        encoding: encoding of both files (a UTF-8 BOM is accepted).

    Returns:
        The enriched hierarchy (also written to the content folder).

    Matching is by title in document order (case/whitespace/inline-markup
    insensitive), so duplicate titles resolve correctly.  Line endings inside
    the text are preserved exactly (CRLF stays ``\\r\\n``).
    """
    # `dir / name` keeps an absolute `name` as-is, so explicit full paths still work.
    hierarchy_path = CODE_SECTION_HIERARCHY_TMP_DIR / hierarchy_file
    markdown_path = ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR / markdown_file
    output_path = CODE_SECTION_CONTENT_TMP_DIR / (output_file or hierarchy_path.name)
    tagged_path = ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_AND_CODE_TAG_DIR / (
        tagged_markdown_file or markdown_path.name
    )
    for dst, src in ((output_path, hierarchy_path), (tagged_path, markdown_path)):
        if dst.resolve() == src.resolve():
            raise ValueError(f"output would overwrite the input file: {dst}")

    enc = "utf-8-sig" if encoding.lower().replace("_", "-") in ("utf-8", "utf8") else encoding
    with open(hierarchy_path, encoding=enc) as fh:
        data = json.load(fh)
    with open(markdown_path, encoding=enc, newline="") as fh:  # newline="" keeps CRLF
        text = fh.read()

    # Accept {"sections": [...]} or a bare list.
    if isinstance(data, list):
        roots, result, root_label = data, None, "root"
    elif isinstance(data, dict) and isinstance(data.get(root_key), list):
        roots, result, root_label = data[root_key], dict(data), root_key
    else:
        raise ValueError(f"hierarchy must be a list or an object with a {root_key!r} list")

    new_roots, flat = _build_skeleton(
        roots, title_key, children_key, content_key, level_key, root_label
    )

    headings = _scan_headings(text)
    starts = [h[1] for h in headings]
    by_title: dict[str, list[int]] = {}
    for idx, h in enumerate(headings):
        by_title.setdefault(h[0], []).append(idx)

    cursor = 0  # headings are consumed in order; also resolves duplicate titles
    missing: list[str] = []
    spans: list[tuple[int, int, str]] = []  # (start, stop, tag), in document order
    seen_tags: set[str] = set()
    for node, title, path in flat:
        hits = by_title.get(_normalize(title))
        j = bisect_left(hits, cursor) if hits else 0
        if not hits or j == len(hits):
            missing.append(title)
            continue  # header_level and content stay None
        idx = hits[j]
        cursor = idx + 1
        _, start, level, body_start = headings[idx]
        end = starts[cursor] if cursor < len(starts) else len(text)

        body = text[body_start:end]  # everything after the heading line(s)
        if trim:
            body = body[_LEADING_BLANK.match(body).end():].rstrip()
        node[level_key] = level
        node[content_key] = body

        # The tagged markdown keeps the heading line(s) and replaces everything from
        # there to the end of the body; blank lines before the next heading stay.
        stop = max(body_start, start + len(text[start:end].rstrip()))
        tag = f"[[content - {path}]]"
        if tag in seen_tags:
            log.warning("duplicate tag %s: the tagged markdown cannot be restored unambiguously", tag)
        seen_tags.add(tag)
        spans.append((body_start, stop, tag))

    if missing:
        msg = f"{len(missing)} title(s) not found in markdown (in order): {missing}"
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    out = new_roots if result is None else {**result, root_key: new_roots}

    # Tagged markdown: one linear pass; spans are disjoint and already sorted.
    nl = "\r\n" if "\r\n" in text else "\n"  # only used if a heading ends the file
    pieces: list[str] = []
    pos = 0
    for body_start, stop, tag in spans:
        pieces.append(text[pos:body_start])  # everything up to and incl. the heading line
        if body_start and text[body_start - 1] not in "\r\n":
            pieces.append(nl)  # heading was the last line, with no line break
        pieces.append(tag)
        if stop == body_start and stop < len(text):
            pieces.append(nl)  # empty body: nothing after the heading ends the tag line
        pos = stop
    pieces.append(text[pos:])

    _atomic_write(output_path, json.dumps(out, ensure_ascii=False, indent=2))
    _atomic_write(tagged_path, "".join(pieces))
    log.info("wrote %s and %s", output_path, tagged_path)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("hierarchy", help="hierarchy JSON file name (in CODE_SECTION_HIERARCHY_TMP_DIR)")
    p.add_argument(
        "markdown", help="markdown file name (in ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR)"
    )
    p.add_argument(
        "-o", "--output", help="output JSON name in CODE_SECTION_CONTENT_TMP_DIR (default: hierarchy name)"
    )
    p.add_argument(
        "-t",
        "--tagged-output",
        help="tagged markdown name in ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_AND_CODE_TAG_DIR "
        "(default: markdown name)",
    )
    p.add_argument("--strict", action="store_true", help="fail if a title is not found")
    p.add_argument("--raw", action="store_true", help="do not trim blank lines / whitespace")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    attach_section_content(
        args.hierarchy,
        args.markdown,
        args.output,
        args.tagged_output,
        strict=args.strict,
        trim=not args.raw,
    )


if __name__ == "__main__":
    main()
