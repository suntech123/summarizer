"""Main entry point orchestrating structural extraction and asset compilation workflow."""

import json
import logging
from config import (
    INPUT_DIR, OUTPUT_MARKDOWN_DIR, EXTRACTED_IMAGES_DIR,
    SUMMARISED_INPUT_MARKDOWN_DIR, FIGURE_ENRICHED_SUMMARY_MARKDOWN_DIR, ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR,
    IMAGE_MAP_TMP_DIR, IMAGE_TAG_TO_ACTUAL_IMAGE_DIR, CODE_SECTION_HIERARCHY_TMP_DIR,
    ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR, CODE_GEN_PROMPT_INSTANCE_DIR,
)
from pdf_utils import convert_pdf_to_markdown
from markdown_utils import extract_figure_image_mapping, enrich_markdown_with_figures, image_lines, img_markers, build_img_map, save_dict, load_dict, replace_img_tags
from section_content import attach_section_content
from code_gen_prompts import build_prompt_instances

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def add_code_tags() -> None:
    for hier in sorted(CODE_SECTION_HIERARCHY_TMP_DIR.glob("*.json")):
        md_name = hier.with_suffix(".md").name        # pairs  x.json <-> x.md
        if not (ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR / md_name).exists():
            logging.warning("no markdown for %s, skipping", hier.name)
            continue
        try:
            attach_section_content(hier.name, md_name, strict=True)
        except (ValueError, OSError) as exc:
            logging.error("%s failed: %s", hier.name, exc)

def generate_code_prompts() -> None:
    try:
        written = build_prompt_instances()   # every *.json in CODE_SECTION_CONTENT_TMP_DIR
    except (ValueError, OSError) as exc:     # bad template/config/JSON, missing files
        logging.error("prompt generation failed: %s", exc)
        return
    logging.info("%d code-generation prompt(s) written to %s",
                 len(written), CODE_GEN_PROMPT_INSTANCE_DIR)

def main() -> None:
    OUTPUT_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{INPUT_DIR.resolve()}'.")
        return

    print(f"Found {len(pdf_files)} PDF file(s) in '{INPUT_DIR}'.")

    for pdf_path in pdf_files:
        doc_name = pdf_path.stem
        markdown_out_path = OUTPUT_MARKDOWN_DIR / f"{doc_name}.md"
        image_dir = EXTRACTED_IMAGES_DIR / doc_name

        print(f"\nConverting '{pdf_path.name}' -> '{markdown_out_path}' (images: '{image_dir}')...")
        convert_pdf_to_markdown(pdf_path, markdown_out_path, image_dir)

        # 1. Extract the figure mapping from the newly generated markdown
        md_content = markdown_out_path.read_text(encoding="utf-8")
        figure_mapping = extract_figure_image_mapping(md_content)
        
        # 2. Check for the corresponding summarized markdown and enrich it
        summary_in_path = SUMMARISED_INPUT_MARKDOWN_DIR / f"{doc_name}.md"
        summary_out_path = FIGURE_ENRICHED_SUMMARY_MARKDOWN_DIR / f"{doc_name}.md"
        
        if summary_in_path.exists():
            enrich_markdown_with_figures(summary_in_path, summary_out_path, figure_mapping)
            print(f"Enriched summary written to '{summary_out_path.resolve()}'.")
        else:
            print(f"Summarized markdown not found at '{summary_in_path}'. Skipping enrichment.")

    print(f"\nDone. {len(pdf_files)} markdown file(s) written to '{OUTPUT_MARKDOWN_DIR.resolve()}'.")
    #for line in image_lines(OUTPUT_MARKDOWN_DIR):
        #print(line)
    #for marker in img_markers(ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR):
        #print(marker)

    # print(list(image_lines(OUTPUT_MARKDOWN_DIR)))
    # print(list(img_markers(ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR)))
    llm_img_lst = list(img_markers(ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR))
    original_img_lst = list(image_lines(OUTPUT_MARKDOWN_DIR))
    if len(llm_img_lst) != len(original_img_lst):
        print(f"Warning: length mismatch: {len(llm_img_lst)} markers vs {len(original_img_lst)} images")
        exit(1)
    img_map = build_img_map(llm_img_lst, original_img_lst)
    #print(f"Image mapping: {json.dumps(img_map, indent=2)}")

    save_dict(img_map, str(IMAGE_MAP_TMP_DIR / f"{doc_name}.img_map.json"))
    img_map_loaded = load_dict(str(IMAGE_MAP_TMP_DIR / f"{doc_name}.img_map.json"))
    print(f"Image mapping: {json.dumps(img_map_loaded, indent=2)}")

    replace_img_tags(str(ORIGINAL_SUMMARY_MARKDOWN_WITH_IMAGE_TAG_DIR), str(IMAGE_TAG_TO_ACTUAL_IMAGE_DIR), str(IMAGE_MAP_TMP_DIR))
    add_code_tags()
    generate_code_prompts()

if __name__ == "__main__":
    main()