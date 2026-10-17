"""Build one code-generation prompt per section from a content JSON and a template.

Reads the section-content JSON written by ``section_content.py`` (nodes with
``title``, ``header_level``, ``content`` and nested ``subsections``) and the prompt
template, then writes one markdown prompt per section that contains a fenced code
block.  Placeholders are filled like this:

    {{MIN_EXAMPLES}}, {{MAX_EXAMPLES}}   -> ``MIN_EXAMPLES`` / ``MAX_EXAMPLES`` from config
    {{LIBRARY_VERSIONS}}                 -> ``LIBRARY_VERSIONS`` from config
    {{EXAMPLE_HEADING_PREFIX}}           -> "#" * (header_level + 1) of the target section
    {{SEC_HIER}}                         -> every section as "#" * header_level + " " + title,
                                            in document order; the target line ends with
                                            " <<< TARGET"
    {{SUB_SEC_CONTENT}}                  -> the target section's ``content``

Folders come from the ``config`` module (same directory as this file); each one
must be a ``pathlib.Path``:
    CODE_SECTION_CONTENT_TMP_DIR   -> content JSON is read from here
    CODE_GEN_PROMPT_TEMPLATE_DIR   -> template is read from here
    CODE_GEN_PROMPT_INSTANCE_DIR   -> prompt files are written here (flat, named by title)

Usage:
    python code_gen_prompts.py [content.json] [--template NAME] [--per-document-folder]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

from config import (
    CODE_GEN_PROMPT_INSTANCE_DIR,
    CODE_GEN_PROMPT_TEMPLATE_DIR,
    CODE_SECTION_CONTENT_TMP_DIR,
    LIBRARY_VERSIONS,
    MAX_EXAMPLES,
    MIN_EXAMPLES,
)

log = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]

DEFAULT_TEMPLATE_NAME = "code-generation-template.md"
TARGET_MARKER = " <<< TARGET"
MAX_FILE_STEM = 120  # keeps Windows paths comfortably below MAX_PATH

# JSON key names written by section_content.py
_ROOT, _CHILDREN, _TITLE, _LEVEL, _CONTENT = (
    "sections",
    "subsections",
    "title",
    "header_level",
    "content",
)

_PLACEHOLDER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_FENCE_OPEN = re.compile(r"^ {0,3}(?:`{3,}|~{3,})", re.M)
_BAD_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"}
_RESERVED_NAMES |= {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


class _Node(NamedTuple):
    title: str
    level: Optional[int]  # None when the node has no usable header_level
    content: Optional[str]


# --------------------------------------------------------------------------- #
# Reading the content JSON
# --------------------------------------------------------------------------- #
def _flatten(data: Any) -> list[_Node]:
    """Return every node in document (pre-order) order.  Iterative: any depth."""
    roots = data.get(_ROOT) if isinstance(data, dict) else data
    if not isinstance(roots, list):
        raise ValueError(f"content JSON must be a list or an object with a {_ROOT!r} list")

    nodes: list[_Node] = []
    stack = [(iter(enumerate(roots)), _ROOT)]
    while stack:
        it, label = stack[-1]
        for i, node in it:
            here = f"{label}[{i}]"
            if not isinstance(node, dict):
                raise ValueError(f"{here}: expected an object, got {type(node).__name__}")
            title = node.get(_TITLE)
            if not isinstance(title, str):
                raise ValueError(f"{here}: missing or non-string {_TITLE!r}")
            level, content = node.get(_LEVEL), node.get(_CONTENT)
            if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 6:
                level = None
            nodes.append(_Node(title, level, content if isinstance(content, str) else None))

            kids = node.get(_CHILDREN)
            if kids is not None and not isinstance(kids, list):
                raise ValueError(f"{here}.{_CHILDREN}: expected a list")
            if kids:
                stack.append((iter(enumerate(kids)), f"{here}.{_CHILDREN}"))
                break  # descend: pre-order == document order
        else:
            stack.pop()
    return nodes


def _has_code(content: str) -> bool:
    """True if the markdown contains a fenced code block (``` or ~~~)."""
    return _FENCE_OPEN.search(content) is not None


# --------------------------------------------------------------------------- #
# Template handling
# --------------------------------------------------------------------------- #
def _lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _config_values() -> dict[str, str]:
    """Validate the config constants and return the static placeholder values."""
    for name, value in (("MIN_EXAMPLES", MIN_EXAMPLES), ("MAX_EXAMPLES", MAX_EXAMPLES)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"config.{name} must be a positive integer, got {value!r}")
    if MIN_EXAMPLES > MAX_EXAMPLES:
        raise ValueError(f"config.MIN_EXAMPLES ({MIN_EXAMPLES}) > MAX_EXAMPLES ({MAX_EXAMPLES})")
    if not isinstance(LIBRARY_VERSIONS, str) or not LIBRARY_VERSIONS.strip():
        raise ValueError("config.LIBRARY_VERSIONS must be a non-empty string")
    return {
        "MIN_EXAMPLES": str(MIN_EXAMPLES),
        "MAX_EXAMPLES": str(MAX_EXAMPLES),
        "LIBRARY_VERSIONS": " ".join(LIBRARY_VERSIONS.split()),
    }


_PER_PROMPT = ("EXAMPLE_HEADING_PREFIX", "SEC_HIER", "SUB_SEC_CONTENT")


def _tokenize(template: str, known: set[str]) -> list[str]:
    """Split the template into literal text (even indexes) and placeholder names (odd).

    Rendering from these tokens is a single pass, so text that looks like a
    placeholder inside a section's content is never substituted again.
    """
    tokens = _PLACEHOLDER.split(template)
    names = set(tokens[1::2])
    unknown = sorted(names - known)
    if unknown:
        raise ValueError(f"template has unknown placeholder(s): {', '.join(unknown)}")
    for name in sorted(known - names):
        log.warning("template does not use {{%s}}", name)
    return tokens


# --------------------------------------------------------------------------- #
# Output files
# --------------------------------------------------------------------------- #
def _file_stem(title: str) -> str:
    """Windows-safe file name stem derived from a section title."""
    stem = _BAD_NAME_CHARS.sub("_", title.replace(":", " -"))
    stem = " ".join(stem.split())[:MAX_FILE_STEM].rstrip(" .")
    if not stem:
        stem = "section"
    if stem.split(".")[0].upper() in _RESERVED_NAMES:
        stem = "_" + stem
    return stem


def _unique_stem(stem: str, used: set[str]) -> str:
    """Make ``stem`` unique (case-insensitively) within this run by adding " (2)", ..."""
    candidate, n = stem, 2
    while candidate.casefold() in used:
        candidate = f"{stem} ({n})"
        n += 1
    used.add(candidate.casefold())
    return candidate


def _atomic_write(path: Path, data: str) -> None:
    """Write text via temp file + rename; always LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _build_for_document(
    json_path: Path,
    tokens: list[str],
    static: dict[str, str],
    out_dir: Path,
    used: set[str],
    encoding: str,
) -> list[Path]:
    with open(json_path, encoding=encoding) as fh:
        nodes = _flatten(json.load(fh))

    # Section hierarchy: only nodes with a usable header_level can be drawn as a heading.
    usable = []
    for node in nodes:
        if node.level is None:
            log.warning("%s: %r has no valid header_level/content, skipped", json_path.name, node.title)
        else:
            usable.append(node)
    lines = ["#" * n.level + " " + " ".join(n.title.split()) for n in usable]

    written: list[Path] = []
    skipped_no_code = 0
    for i, node in enumerate(usable):
        if node.content is None or not _has_code(node.content):
            skipped_no_code += 1
            continue
        if node.level == 6:
            log.warning("%r is a level-6 heading: example heading prefix is capped at 6 hashes", node.title)

        values = {
            **static,
            "EXAMPLE_HEADING_PREFIX": "#" * min(node.level + 1, 6),
            "SEC_HIER": "\n".join(lines[:i] + [lines[i] + TARGET_MARKER] + lines[i + 1:]),
            "SUB_SEC_CONTENT": _lf(node.content).strip("\n"),
        }
        prompt = "".join(t if k % 2 == 0 else values[t] for k, t in enumerate(tokens))

        path = out_dir / (_unique_stem(_file_stem(node.title), used) + ".md")
        _atomic_write(path, prompt.rstrip("\n") + "\n")
        written.append(path)
        log.info("wrote %s", path)

    log.info(
        "%s: %d prompt(s) written, %d section(s) skipped (no fenced code)",
        json_path.name,
        len(written),
        skipped_no_code,
    )
    return written


def build_prompt_instances(
    json_name: Optional[PathLike] = None,
    template_name: PathLike = DEFAULT_TEMPLATE_NAME,
    *,
    per_document_folder: bool = False,
    encoding: str = "utf-8",
) -> list[Path]:
    """Write one prompt per code-bearing section into CODE_GEN_PROMPT_INSTANCE_DIR.

    Args:
        json_name: content JSON inside CODE_SECTION_CONTENT_TMP_DIR.  ``None``
            processes every ``*.json`` there.  An absolute path is used as-is.
        template_name: template file inside CODE_GEN_PROMPT_TEMPLATE_DIR.
        per_document_folder: put each JSON's prompts in a subfolder named after
            the JSON file (avoids clashes between documents with identical
            section titles).  Default: one flat folder.
        encoding: encoding of the JSON and the template (a UTF-8 BOM is accepted).

    Returns:
        The paths of the prompt files written, in document order.

    Files are named after the section title (``4.1.1 Adding color channels.md``;
    characters Windows forbids are replaced, a colon becomes `` -``).  Two titles
    that map to the same name in one run get `` (2)``, `` (3)``, ...; a later run
    overwrites earlier files of the same name.
    """
    enc = "utf-8-sig" if encoding.lower().replace("_", "-") in ("utf-8", "utf8") else encoding
    static = _config_values()

    with open(CODE_GEN_PROMPT_TEMPLATE_DIR / template_name, encoding=enc, newline="") as fh:
        template = _lf(fh.read())
    tokens = _tokenize(template, set(static) | set(_PER_PROMPT))
    if "SUB_SEC_CONTENT" not in tokens[1::2]:
        raise ValueError("template must contain {{SUB_SEC_CONTENT}}")

    if json_name is None:
        json_paths = sorted(CODE_SECTION_CONTENT_TMP_DIR.glob("*.json"))
        if not json_paths:
            log.warning("no *.json files found in %s", CODE_SECTION_CONTENT_TMP_DIR)
    else:
        json_paths = [CODE_SECTION_CONTENT_TMP_DIR / json_name]

    used: set[str] = set()  # file names taken in this run (case-insensitive)
    written: list[Path] = []
    for json_path in json_paths:
        out_dir = CODE_GEN_PROMPT_INSTANCE_DIR
        if per_document_folder:
            out_dir = out_dir / json_path.stem
            used.clear()  # names only need to be unique within a folder
        written += _build_for_document(json_path, tokens, static, out_dir, used, enc)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Build one code-generation prompt per section.")
    p.add_argument(
        "json",
        nargs="?",
        help="content JSON name in CODE_SECTION_CONTENT_TMP_DIR (default: every *.json there)",
    )
    p.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE_NAME,
        help=f"template name in CODE_GEN_PROMPT_TEMPLATE_DIR (default: {DEFAULT_TEMPLATE_NAME})",
    )
    p.add_argument(
        "--per-document-folder",
        action="store_true",
        help="write each JSON's prompts into a subfolder named after the JSON file",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_prompt_instances(
        args.json, args.template, per_document_folder=args.per_document_folder
    )


if __name__ == "__main__":
    main()