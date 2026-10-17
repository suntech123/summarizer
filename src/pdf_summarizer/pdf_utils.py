"""High-level bindings for PyMuPDF compilation and asset isolation operations."""

import shutil
from pathlib import Path
import pymupdf
import pymupdf4llm
from markdown_pdf import MarkdownPdf, Section
from config import IMAGE_CSS, IMAGE_WIDTH_PT, IMAGE_DPI, IMAGE_FORMAT
from markdown_utils import split_at_images, escape_literal_hashes

# def convert_pdf_to_markdown(pdf_path: Path, markdown_out_path: Path, image_dir: Path) -> None:
#     """Convert a single PDF into a markdown file, writing its images into image_dir."""
#     image_dir.mkdir(parents=True, exist_ok=True)
#     markdown_out_path.parent.mkdir(parents=True, exist_ok=True)

#     pages = pymupdf4llm.to_markdown(
#         str(pdf_path),
#         write_images=True,
#         image_path=str(image_dir),
#         force_text=False,
#         page_chunks=True,
#     )

#     markdown_out_path.write_text(escape_literal_hashes(pages), encoding="utf-8")


def convert_pdf_to_markdown(pdf_path: Path, markdown_out_path: Path, image_dir: Path,
                            dpi: int = IMAGE_DPI) -> None:
    """Convert a single PDF into a markdown file, writing its images into image_dir."""
    shutil.rmtree(image_dir, ignore_errors=True)  # remove images from earlier (150 dpi) runs
    image_dir.mkdir(parents=True, exist_ok=True)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)

    pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        write_images=True,
        image_path=str(image_dir),
        image_format=IMAGE_FORMAT,
        dpi=dpi,
        force_text=False,
        page_chunks=True,
    )

    markdown_out_path.write_text(escape_literal_hashes(pages), encoding="utf-8")

def render_pdf(markdown_content: str, out_path: Path, max_passes: int = 6) -> None:
    """Render markdown to PDF dynamically.

    Inserts strategic page breaks before any images that risk scaling down
    due to limited layout space on the target page bounds.
    """

    parts = split_at_images(markdown_content)
    break_before: set[int] = set()

    for _ in range(max_passes):
        chunks: list[str] = []
        current = parts[0]

        for i, part in enumerate(parts[1:]):
            if i in break_before:
                chunks.append(current)
                current = part
            else:
                current += part
        chunks.append(current)

        pdf = MarkdownPdf()
        for chunk in chunks:
            pdf.add_section(Section(chunk, toc=False, root="."), user_css=IMAGE_CSS)
        pdf.save(str(out_path))

        squeezed: set[int] = set()
        index = 0

        # Validate layout compression artifacts
        with pymupdf.open(str(out_path)) as doc:
            for page in doc:
                for img in page.get_images(full=True):
                    if page.get_image_bbox(img).width < IMAGE_WIDTH_PT - 2:
                        squeezed.add(index)
                    index += 1

        if not squeezed:
            break
        break_before |= squeezed

def extract_raw_images(pdf_path: Path, output_dir: Path) -> None:
    """Extract all embedded binary image objects directly from the PDF source."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(str(pdf_path)) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                filepath = output_dir / filename

                filepath.write_bytes(image_bytes)

        print(f"Extraction complete! Images saved to: '{output_dir.resolve()}'")


