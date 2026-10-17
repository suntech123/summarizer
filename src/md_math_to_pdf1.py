#!/usr/bin/env python3
r"""
md_math_to_pdf.py - normalise Markdown that contains LaTeX math, then render a PDF.

Why this exists
---------------
Markdown files exported from chat tools / notes apps / OCR often write math as

    inline:   \( x^2 \)     or   \\( x^2 \\)
    display:  \[ ... \]     or   \\[ ... \\]

Pandoc (and most Markdown renderers) only recognise $...$ and $$...$$ by default,
so those equations come out as literal text. This script rewrites the math
delimiters, cleans up a few other things that trip up LaTeX, and then calls
pandoc to build the PDF.

What it does
------------
1. Protects code blocks / inline code so nothing inside them is touched.
2. Converts \( \) -> $ $ and \[ \] -> $$ $$ (single- or double-backslash forms),
   trims stray whitespace, and removes blank lines inside display math
   (a blank line inside math is a hard LaTeX error).
3. Tidies structure (can be switched off with --no-heading-fix):
     * strips **bold** wrappers inside '#' headings
     * merges "Chapter N" + title headings into one "Chapter N: Title"
     * promotes stand-alone bold lines like **2.1 Overview** / **Notes** to real
       headings (2.1 -> level 2, 2.2.1 -> level 3)
     * re-bases heading levels so the shallowest heading is '#'
     * drops '---' separator lines (use --keep-rules to keep them)
     * styles "Figure 2.1 - caption..." paragraphs as small indented captions
4. Removes emoji (LaTeX fonts have no glyphs for them) and lines matching the
   default --drop pattern (a trailing "💡 Would you like..." chat prompt).
5. Runs pandoc -> PDF (xelatex by default).

Requirements
------------
    pandoc  +  a TeX distribution (TeX Live / MacTeX / MiKTeX) providing xelatex
    Python 3.8+  (standard library only)

Usage
-----
    python md_math_to_pdf.py chapter2.md                  # -> chapter2.pdf
    python md_math_to_pdf.py chapter2.md -o notes.pdf --toc
    python md_math_to_pdf.py chapter2.md --emit-md        # only write chapter2.normalized.md
    python md_math_to_pdf.py chapter2.md --html           # MathML HTML (no LaTeX needed)
    python md_math_to_pdf.py chapter2.md --drop "^Notes:.*$"   # extra lines to remove

    python md_math_to_pdf.py chapter2.md --no-heading-fix      # leave headings exactly as written
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #

FENCED_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ \t]*(?P=fence)[ \t]*$",
    re.S | re.M,
)
INLINE_CODE_RE = re.compile(r"(`+)(?!`)[^\n]*?(?<!`)\1(?!`)")
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n(?:---|\.\.\.)[ \t]*\n", re.S)

# \( ... \)  /  \\( ... \\)   and   \[ ... \]  /  \\[ ... \\]
MATH_RE = re.compile(
    r"(?P<disp>\\{1,2}\[(?P<d>.+?)\\{1,2}\])"
    r"|(?P<inl>\\{1,2}\((?P<i>.+?)\\{1,2}\))",
    re.S,
)

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\uFE0F\u200D]+"
)

ATX_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
RULE_RE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
BOLD_NUMBERED_RE = re.compile(r"^\*\*(\d+(?:\.\d+)+)[ \t]+(.+?)\*\*[ \t]*$")
BOLD_ONLY_RE = re.compile(r"^\*\*([^*\n]{1,60}?)\*\*[ \t]*$")
CHAPTER_RE = re.compile(r"^(?:Chapter|Part|Appendix|Unit|Module)[ \t]+[\w.]+$", re.I)
FIGURE_RE = re.compile(r"^(Figure|Fig\.|Table)[ \t]+(\d+(?:\.\d+)*)[ \t]*[-\u2013\u2014:.][ \t]+(.*)$")

DEFAULT_DROP = [r"^[ \t]*\U0001F4A1.*$"]  # '💡 Would you like to create a study guide...'

# amsmath's \boldsymbol clashes with unicode-math (the math package pandoc loads for
# xelatex/lualatex) and aborts with "Improper alphabetic constant". Map it to the
# bold-italic math alphabet instead; the user's equations stay untouched.
UNICODE_MATH_PREAMBLE = r"""
\AtBeginDocument{\renewcommand{\boldsymbol}[1]{\symbfit{#1}}}
"""


@dataclass
class Heading:
    level: int
    text: str


# --------------------------------------------------------------------------- #
# Step 1 - protect code so we never rewrite inside it
# --------------------------------------------------------------------------- #

def protect_code(text: str) -> tuple[str, list[str]]:
    stash: list[str] = []

    def keep(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"\x00CODE{len(stash) - 1}\x00"

    text = FENCED_RE.sub(keep, text)
    text = INLINE_CODE_RE.sub(keep, text)
    return text, stash


def restore_code(text: str, stash: list[str]) -> str:
    return re.sub(r"\x00CODE(\d+)\x00", lambda m: stash[int(m.group(1))], text)


# --------------------------------------------------------------------------- #
# Step 2 - math delimiters
# --------------------------------------------------------------------------- #

def _clean_display(body: str) -> str:
    lines = [ln.rstrip() for ln in body.strip().splitlines() if ln.strip()]
    return textwrap.dedent("\n".join(lines)).strip()


def convert_math(text: str) -> tuple[str, int, int]:
    n_inline = n_display = 0

    def repl(m: re.Match) -> str:
        nonlocal n_inline, n_display
        if m.group("disp"):
            body = _clean_display(m.group("d"))
            if not body:
                return m.group(0)
            n_display += 1
            return f"\n\n$$\n{body}\n$$\n\n"
        body = " ".join(m.group("i").split())
        if not body:
            return m.group(0)
        n_inline += 1
        return f"${body}$"

    text = MATH_RE.sub(repl, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, n_inline, n_display


# --------------------------------------------------------------------------- #
# Step 3 - document structure
# --------------------------------------------------------------------------- #

def _strip_bold(s: str) -> str:
    return re.sub(r"(\*\*|__)(.+?)\1", r"\2", s).strip()


def fix_structure(text: str, *, keep_rules: bool, fix_headings: bool, latex: bool) -> str:
    lines = text.split("\n")
    items: list[object] = []

    def blank(i: int) -> bool:
        return i < 0 or i >= len(lines) or not lines[i].strip()

    for i, line in enumerate(lines):
        if fix_headings:
            m = ATX_RE.match(line)
            if m:
                items.append(Heading(len(m.group(1)), _strip_bold(m.group(2))))
                continue
            standalone = blank(i - 1) and blank(i + 1)
            if standalone:
                m = BOLD_NUMBERED_RE.match(line)
                if m:
                    level = min(6, 1 + m.group(1).count("."))
                    items.append(Heading(level, f"{m.group(1)} {m.group(2).strip()}"))
                    continue
                m = BOLD_ONLY_RE.match(line)
                if m and not m.group(1).rstrip().endswith((".", ":", ";", ",", "?", "!")):
                    items.append(Heading(2, m.group(1).strip()))
                    continue
        if not keep_rules and RULE_RE.match(line) and blank(i - 1):
            continue
        m = FIGURE_RE.match(line)
        if m and blank(i - 1) and blank(i + 1):
            caption = f"**{m.group(1)} {m.group(2)}.** {m.group(3)}"
            if latex:
                items.append(
                    "```{=latex}\n\\begin{quote}\\small\n```\n\n" + caption +
                    "\n\n```{=latex}\n\\end{quote}\n```"
                )
            else:
                items.append(caption)
            continue
        items.append(line)

    if fix_headings:
        # "Chapter 2" directly followed by another heading -> one heading
        merged: list[object] = []
        i = 0
        while i < len(items):
            it = items[i]
            if isinstance(it, Heading) and CHAPTER_RE.match(it.text):
                j = i + 1
                while j < len(items) and isinstance(items[j], str) and not items[j].strip():
                    j += 1
                if j < len(items) and isinstance(items[j], Heading):
                    merged.append(Heading(1, f"{it.text}: {items[j].text}"))
                    i = j + 1
                    continue
            merged.append(it)
            i += 1
        items = merged

        # shallowest heading becomes '#'
        levels = [it.level for it in items if isinstance(it, Heading)]
        shift = (min(levels) - 1) if levels else 0
        for it in items:
            if isinstance(it, Heading):
                it.level -= shift

    out = [f"{'#' * it.level} {it.text}" if isinstance(it, Heading) else it for it in items]
    text = "\n".join(out)
    # make sure headings are surrounded by blank lines
    text = re.sub(r"(?m)^(#{1,6} .*)$", r"\n\1\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


# --------------------------------------------------------------------------- #
# Full normalisation pipeline
# --------------------------------------------------------------------------- #

def normalize(
    text: str,
    *,
    latex: bool = True,
    fix_headings: bool = True,
    keep_rules: bool = False,
    keep_emoji: bool = False,
    drop: list[str] | None = None,
) -> tuple[str, dict]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    front = ""
    m = FRONT_MATTER_RE.match(text)
    if m:
        front, text = m.group(0), text[m.end():]

    text, stash = protect_code(text)
    
    ## --- center image start
    if latex:
        text = center_images(text)
    ## --- center image end

    for pat in DEFAULT_DROP + (drop or []):
        text = re.sub(pat, "", text, flags=re.M)

    text, n_inline, n_display = convert_math(text)

    if not keep_emoji:
        text = EMOJI_RE.sub("", text)

    text = fix_structure(text, keep_rules=keep_rules, fix_headings=fix_headings, latex=latex)
    text = restore_code(text, stash)

    stats = {"inline_math": n_inline, "display_math": n_display}
    return front + text.strip() + "\n", stats


# --------------------------------------------------------------------------- #
# Rendering via pandoc
# --------------------------------------------------------------------------- #

def _tex_file_exists(name: str) -> bool:
    """True if kpsewhich can find `name`, or if kpsewhich is unavailable (assume yes)."""
    if not shutil.which("kpsewhich"):
        return True
    res = subprocess.run(["kpsewhich", name], capture_output=True, text=True)
    return bool(res.stdout.strip())


def run_pandoc(md_path: Path, out_path: Path, *, html: bool, engine: str,
               toc: bool, title: str | None, resource_dir: Path) -> None:
    if not shutil.which("pandoc"):
        sys.exit("error: pandoc not found. Install it from https://pandoc.org/installing.html")

    cmd = ["pandoc", str(md_path), "-f", "markdown+tex_math_dollars+raw_tex+smart",
           "-o", str(out_path), f"--resource-path={resource_dir}"]
    if toc:
        cmd.append("--toc")
    if title:
        cmd += ["-M", f"title={title}"]

    if html:
        cmd += ["-t", "html5", "--standalone", "--mathml", "-M", "pagetitle=" + (title or out_path.stem)]
    else:
        if not shutil.which(engine):
            sys.exit(f"error: '{engine}' not found. Install a TeX distribution "
                     "(TeX Live / MacTeX / MiKTeX), or use --html.")
        cmd += [f"--pdf-engine={engine}",
                "-V", "geometry:margin=1in", "-V", "fontsize=11pt",
                "-V", "colorlinks=true", "-V", "linkcolor=blue"]
        # pandoc's LaTeX template loads `lmodern` unless `fontfamily` is set. Minimal TeX
        # installs (e.g. Debian's texlive-latex-base only) lack lmodern.sty, which aborts
        # the build. For Unicode engines fontspec already defaults to Latin Modern, so
        # setting fontfamily to a harmless, always-present package (`bm`) is safe.
        if engine != "pdflatex" and not _tex_file_exists("lmodern.sty"):
            cmd += ["-V", "fontfamily=bm"]
        if engine != "pdflatex":
            header = md_path.with_name("header.tex")
            header.write_text(UNICODE_MATH_PREAMBLE, encoding="utf-8")
            cmd += ["-H", str(header)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-3000:] + "\n")
        sys.exit("error: pandoc failed (see above). Re-run with --emit-md to inspect the "
                 "normalised Markdown.")
    if proc.stderr.strip():
        sys.stderr.write(proc.stderr)

##### Start -- Added for centering image
IMG_LINE_RE = re.compile(r"^[ \t]*(!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?)[ \t]*$", re.M)

def center_images(text: str) -> str:
    """Wrap stand-alone image lines in a LaTeX center environment (PDF output only)."""
    return IMG_LINE_RE.sub(
        lambda m: "```{=latex}\n\\begin{center}\n```\n" + m.group(1) + "\n```{=latex}\n\\end{center}\n```",
        text,
    )
##### End -- Added for centering image

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Normalise math Markdown and render it to PDF.")
    ap.add_argument("input", type=Path, help="Markdown file")
    ap.add_argument("-o", "--output", type=Path, help="output file (default: <input>.pdf)")
    ap.add_argument("--engine", default="xelatex", choices=["xelatex", "lualatex", "pdflatex"])
    ap.add_argument("--toc", action="store_true", help="add a table of contents")
    ap.add_argument("--title", help="document title (metadata)")
    ap.add_argument("--html", action="store_true", help="write standalone HTML with MathML instead of PDF")
    ap.add_argument("--emit-md", nargs="?", const="", metavar="PATH",
                    help="only write the normalised Markdown (default: <input>.normalized.md)")
    ap.add_argument("--keep-rules", action="store_true", help="keep '---' separator lines")
    ap.add_argument("--keep-emoji", action="store_true", help="do not strip emoji")
    ap.add_argument("--no-heading-fix", action="store_true", help="leave headings exactly as written")
    ap.add_argument("--drop", action="append", default=[], metavar="REGEX",
                    help="delete lines matching REGEX (repeatable)")
    args = ap.parse_args(argv)

    if not args.input.is_file():
        sys.exit(f"error: {args.input} not found")

    src = args.input.read_text(encoding="utf-8")
    emit_only = args.emit_md is not None
    md, stats = normalize(
        src,
        latex=not (args.html or emit_only),
        fix_headings=not args.no_heading_fix,
        keep_rules=args.keep_rules,
        keep_emoji=args.keep_emoji,
        drop=args.drop,
    )
    print(f"converted {stats['inline_math']} inline and {stats['display_math']} display equations")

    if emit_only:
        out = Path(args.emit_md) if args.emit_md else args.input.with_suffix(".normalized.md")
        out.write_text(md, encoding="utf-8")
        print(f"wrote {out}")
        return

    out = args.output or args.input.with_suffix(".html" if args.html else ".pdf")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_md = Path(tmp) / "normalized.md"
        tmp_md.write_text(md, encoding="utf-8")
        run_pandoc(tmp_md, out, html=args.html, engine=args.engine, toc=args.toc,
                   title=args.title, resource_dir=args.input.resolve().parent)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
