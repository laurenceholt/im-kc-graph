#!/usr/bin/env python3
"""
Extract individual math assessment questions from Illustrative Mathematics (IM) PDFs
and store them in a PostgreSQL database or export as a static site.

Usage:
    # Preview mode - save images locally to check extraction quality
    python extract_questions.py --preview

    # Store in database
    python extract_questions.py

    # Process all PDFs (default: first only)
    python extract_questions.py --all

    # Export static site (processes all PDFs, outputs to site/)
    python extract_questions.py --export-site

    # Export for a specific unit (multi-unit pipeline)
    python extract_questions.py --export-site --pdf-dir "path/to/PDFs" --unit G5_U1
"""

import fitz  # PyMuPDF
import re
import os
import sys
import io
import json
import argparse
import logging
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DPI = 200
SCALE = DPI / 72.0
HEADER_MARGIN_PT = 45
FOOTER_MARGIN_PT = 55
QUESTION_PADDING_PT = 3
PDF_DIR = "IM Assessments"
OUTPUT_DIR = "extracted_questions"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS questions (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    grade TEXT NOT NULL,
    unit TEXT NOT NULL,
    section TEXT,
    assessment_number INTEGER NOT NULL,
    question_number INTEGER NOT NULL,
    image_data BYTEA NOT NULL,
    image_width INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    source_pages INTEGER[] NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(filename, assessment_number, question_number)
);
"""


def setup_logging(unit=None):
    """Set up logging to both console and a timestamped log file.

    Returns the log file path.
    """
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unit_part = f"_{unit}" if unit else ""
    log_file = os.path.join(log_dir, f'extract{unit_part}_{timestamp}.log')

    logger = logging.getLogger('extract')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — captures everything (DEBUG+)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    # Console handler — INFO+
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return log_file


log = logging.getLogger('extract')


def parse_filename(filepath):
    """Extract grade, unit, section from the PDF filename.

    Supports multiple IM filename patterns. The parser is intentionally
    flexible to handle variations as more PDFs are provided.

    Patterns tried (in order):
      1. IM_G5_U1_SA_...pdf  (structured: grade, unit, section)
      2. IM_G5_U1_...pdf     (structured: grade, unit, no section)
      3. Grade5-8-...pdf      (loose: infer grade from name)

    Grade mapping for high school courses:
      A1 -> G9 (Algebra 1), GEO -> G10 (Geometry), A2 -> G11 (Algebra 2)
      K  -> GK (Kindergarten)
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]

    # Pattern 1: Structured IM naming — IM_{grade}_{unit}[_{section}]_...
    match = re.match(
        r'IM_((?:G|GK|A|GEO)\w*)_(U\d+)(?:_(S[A-Z]))?',
        basename
    )
    if match:
        grade_map = {
            'A1': 'G9', 'A2': 'G11', 'GEO': 'G10', 'K': 'GK',
        }
        raw_grade = match.group(1)
        grade = grade_map.get(raw_grade, raw_grade)
        return {
            'grade': grade,
            'unit': match.group(2),
            'section': match.group(3),
        }

    # Pattern 2: Loose naming — try to extract grade from "Grade5" or "G5" in filename
    m = re.search(r'Grade[\s-]*(\d+)', basename, re.IGNORECASE)
    if not m:
        m = re.search(r'(?:^|[_-])G(\d+)(?:[_-]|$)', basename)
    if m:
        grade_num = m.group(1)
        grade = f"G{grade_num}"
        # Try to find unit number
        u = re.search(r'(?:Unit|U)[\s-]*(\d+)', basename, re.IGNORECASE)
        unit = f"U{u.group(1)}" if u else "U0"
        # Try to find section
        s = re.search(r'(?:Section|S)[\s-]*([A-Z])', basename, re.IGNORECASE)
        section = f"S{s.group(1).upper()}" if s else None
        return {
            'grade': grade,
            'unit': unit,
            'section': section,
        }

    raise ValueError(f"Could not parse metadata from filename: {basename}")


def analyze_page(page):
    """Analyze a PDF page to find question start markers and assessment info.

    Detects both single-column and two-column (side-by-side) question layouts.
    Right-column markers are only accepted if paired with a left-column marker
    at a similar y position.
    """
    text_dict = page.get_text("dict")
    blocks = text_dict.get("blocks", [])

    left_markers = []   # [(question_num, y_top_pt)]
    right_markers = []  # [(question_num, y_top_pt, x_left)]
    assessment_num = None
    banner_bottom_pt = None
    page_width = page.rect.width
    page_midpoint = page_width / 2

    for block in blocks:
        if block.get("type") != 0:  # text blocks only
            continue

        for line in block["lines"]:
            spans = line.get("spans", [])
            if not spans:
                continue

            full_text = "".join(s["text"] for s in spans).strip()
            line_bbox = line["bbox"]

            # Detect assessment number from page header
            # IM pattern: "End-of-Course Assessment" or "Unit Assessment N"
            m = re.search(r'(?:Unit|End.of.Course)\s*Assessment\s*(\d+)?', full_text, re.IGNORECASE)
            if m:
                assessment_num = int(m.group(1)) if m.group(1) else 1

            # Detect ANSWER KEY banner (only in top half of page, not footer)
            if (re.search(r'ANSWER\s*KEY', full_text, re.IGNORECASE)
                    and line_bbox[1] < page.rect.height / 2):
                banner_bottom_pt = line_bbox[3] + 10
                m2 = re.search(r'ASSESSMENT[- ]*(\d+)', full_text, re.IGNORECASE)
                if m2 and assessment_num is None:
                    assessment_num = int(m2.group(1))

            # Detect question start: "N. " or "N."
            # Skip header/footer zone (top 40pt and bottom 50pt)
            page_height = page.rect.height
            if line_bbox[1] < 40 or line_bbox[1] > page_height - 50:
                continue

            m = re.match(r'^(\d+)\.\s', full_text) or re.fullmatch(r'(\d+)\.', full_text)
            if m:
                q_num = int(m.group(1))
                x_left = line_bbox[0]

                if x_left < 140:
                    # Left column marker (standard)
                    left_markers.append((q_num, line_bbox[1]))
                elif page_midpoint - 20 < x_left < page_midpoint + 80:
                    # Potential right column marker (near page center)
                    right_markers.append((q_num, line_bbox[1], x_left))

    # Build final marker list: left markers are always included.
    # Right markers only if paired with a left marker at similar y (within 30pt).
    question_markers = [(q, y, 'left') for q, y in left_markers]

    paired_right = []
    for rq, ry, rx in right_markers:
        for lq, ly in left_markers:
            if abs(ly - ry) < 30:
                paired_right.append((rq, ry, 'right'))
                break

    question_markers.extend(paired_right)

    return {
        'question_markers': sorted(question_markers, key=lambda x: x[1]),
        'assessment_num': assessment_num,
        'banner_bottom_pt': banner_bottom_pt,
    }


def extract_question_image(page, top_pt, bottom_pt, left_pt=0, right_pt=None):
    """Render a cropped region of a page as a PNG image."""
    if right_pt is None:
        right_pt = page.rect.width
    # Ensure valid clip region (minimum 10pt height)
    if bottom_pt - top_pt < 10:
        bottom_pt = top_pt + 10
    clip = fitz.Rect(left_pt, top_pt, right_pt, bottom_pt)
    clip = clip & page.rect  # intersect with page bounds
    if clip.is_empty or clip.width < 1 or clip.height < 1:
        raise ValueError(f"Invalid clip region: {clip}")
    pix = page.get_pixmap(dpi=DPI, clip=clip)
    return pix.tobytes("png"), pix.width, pix.height


def stitch_images_vertically(image_data_list):
    """Stitch multiple PNG images vertically into one."""
    if len(image_data_list) == 1:
        data = image_data_list[0]
        img = Image.open(io.BytesIO(data))
        return data, img.width, img.height

    images = [Image.open(io.BytesIO(d)) for d in image_data_list]
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    stitched = Image.new("RGB", (max_width, total_height), "white")
    y_offset = 0
    for img in images:
        stitched.paste(img, (0, y_offset))
        y_offset += img.height

    buf = io.BytesIO()
    stitched.save(buf, format="PNG")
    return buf.getvalue(), stitched.width, stitched.height


def trim_whitespace(png_data, padding=20, threshold=250):
    """Trim excess whitespace from the bottom of a PNG image.

    Scans the center 80% of each row to ignore edge decorations (sidebars, tabs).
    Keeps content + padding pixels.
    """
    img = Image.open(io.BytesIO(png_data)).convert("RGB")
    width, height = img.size

    # Ignore edge decorations: only scan center 80% of width
    margin = int(width * 0.1)
    scan_left = margin
    scan_right = width - margin

    # Scan from bottom to find last non-white row
    import numpy as np
    arr = np.array(img)
    last_content_row = 0
    for y in range(height - 1, -1, -1):
        row_pixels = arr[y, scan_left:scan_right, :]
        if np.any(row_pixels < threshold):
            last_content_row = y
            break

    # Crop with padding
    crop_bottom = min(height, last_content_row + padding)
    if crop_bottom < height - 10:  # only trim if saving meaningful space
        img = img.crop((0, 0, width, crop_bottom))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.width, img.height


def extract_questions_from_pdf(pdf_path):
    """Extract all individual question images from a PDF."""
    doc = fitz.open(pdf_path)
    metadata = parse_filename(pdf_path)

    # First pass: analyze every page
    page_analyses = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        analysis = analyze_page(page)
        analysis['page_num'] = page_num
        analysis['page_height'] = page.rect.height
        page_analyses.append(analysis)

    # Forward-fill assessment numbers across pages
    current_assessment = None
    for analysis in page_analyses:
        if analysis['assessment_num'] is not None:
            current_assessment = analysis['assessment_num']
        analysis['resolved_assessment'] = current_assessment

    # Section-level PDFs don't have "Unit Assessment N" headers.
    # They contain a single assessment, so default to 1.
    if current_assessment is None:
        for analysis in page_analyses:
            analysis['resolved_assessment'] = 1

    # Build question regions: map (assessment, question) -> list of (page, top, bottom, left, right)
    # left/right are horizontal clip bounds (0 and page_width for full-width, or half-page for columns)
    questions_map = {}
    last_question_key = None

    for analysis in page_analyses:
        page_num = analysis['page_num']
        page = doc[page_num]
        page_height = analysis['page_height']
        page_width = page.rect.width
        page_midpoint = page_width / 2
        markers = analysis['question_markers']  # [(q_num, y, column)]
        assessment = analysis['resolved_assessment']

        # Content area (excluding header/footer)
        content_top = HEADER_MARGIN_PT
        if analysis['banner_bottom_pt'] is not None:
            content_top = max(content_top, analysis['banner_bottom_pt'])
        content_bottom = page_height - FOOTER_MARGIN_PT

        # Pages with a banner start a new assessment — never continue previous question
        has_banner = analysis['banner_bottom_pt'] is not None

        # Separate left-only markers from right markers for column handling
        left_markers = [(q, y, col) for q, y, col in markers if col == 'left']
        right_markers = [(q, y, col) for q, y, col in markers if col == 'right']

        if not markers:
            # No new questions on this page — continuation of previous question
            # But don't continue across assessment boundaries (banner pages)
            if (last_question_key and last_question_key in questions_map
                    and not has_banner):
                questions_map[last_question_key]['regions'].append(
                    (page_num, content_top, content_bottom, 0, page_width)
                )
            continue

        # If there's content above the first question marker on this page,
        # it belongs to the previous question (continuation from prior page).
        # Skip if this page has a banner (new assessment) or if the previous
        # question belongs to a different assessment.
        first_marker_y = markers[0][1]
        if (last_question_key and last_question_key in questions_map
                and not has_banner
                and last_question_key[0] == assessment):
            gap = first_marker_y - content_top
            if gap > 80:  # substantial content, not just header whitespace
                questions_map[last_question_key]['regions'].append(
                    (page_num, content_top, first_marker_y - QUESTION_PADDING_PT, 0, page_width)
                )

        # Find the y boundary where side-by-side columns end
        # (the next left-column marker well below the right-column markers)
        column_bottom_y = None
        if right_markers:
            max_right_y = max(ry for _, ry, _ in right_markers)
            # Use tolerance of 30pt to skip markers at the same vertical position
            below_markers = [(q, y) for q, y, col in left_markers if y > max_right_y + 30]
            if below_markers:
                column_bottom_y = below_markers[0][1]
            else:
                column_bottom_y = content_bottom

        for i, (q_num, y_start, column) in enumerate(markers):
            crop_top = max(content_top, y_start - QUESTION_PADDING_PT)

            if column == 'right':
                # Right-column question: clip right half, bottom is column_bottom_y
                crop_bottom = column_bottom_y
                left_bound = page_midpoint
                right_bound = page_width
            elif column == 'left' and right_markers and y_start <= max(ry for _, ry, _ in right_markers) + 30:
                # Left-column question that's paired with a right-column one:
                # clip left half, bottom is column_bottom_y
                crop_bottom = column_bottom_y
                left_bound = 0
                right_bound = page_midpoint
            else:
                # Full-width question (standard case)
                left_bound = 0
                right_bound = page_width
                # Find bottom: next marker's y (only among full-width or left markers below columns)
                next_markers = [(q, y) for q, y, c in markers[i+1:]
                                if c == 'left' and (not right_markers or y > max(ry for _, ry, _ in right_markers) + 30)]
                if next_markers:
                    crop_bottom = next_markers[0][1]
                else:
                    crop_bottom = content_bottom

            key = (assessment, q_num)
            questions_map[key] = {
                'assessment_num': assessment,
                'question_num': q_num,
                'regions': [(page_num, crop_top, crop_bottom, left_bound, right_bound)],
            }
            last_question_key = key

    # Render images for each question
    results = []
    for key in sorted(questions_map.keys()):
        q = questions_map[key]
        image_parts = []
        source_pages = []

        for region in q['regions']:
            page_num, top_pt, bottom_pt = region[0], region[1], region[2]
            left_pt = region[3] if len(region) > 3 else 0
            right_pt = region[4] if len(region) > 4 else None
            page = doc[page_num]
            if bottom_pt <= top_pt:
                log.warning(f"    SKIP: Assessment {q['assessment_num']} Q{q['question_num']} "
                            f"page {page_num+1}: invalid region top={top_pt:.1f} bottom={bottom_pt:.1f}")
                continue
            img_data, w, h = extract_question_image(page, top_pt, bottom_pt, left_pt, right_pt)
            image_parts.append(img_data)
            source_pages.append(page_num + 1)  # 1-indexed

        final_data, final_w, final_h = stitch_images_vertically(image_parts)

        # Trim excess whitespace from bottom of image
        final_data, final_w, final_h = trim_whitespace(final_data)

        results.append({
            'filename': os.path.basename(pdf_path),
            'grade': metadata['grade'],
            'unit': metadata['unit'],
            'section': metadata['section'],
            'assessment_number': q['assessment_num'],
            'question_number': q['question_num'],
            'image_data': final_data,
            'image_width': final_w,
            'image_height': final_h,
            'source_pages': source_pages,
        })

    doc.close()
    return results


def save_preview(questions, pdf_basename):
    """Save extracted question images locally for inspection."""
    subdir = os.path.join(OUTPUT_DIR, os.path.splitext(pdf_basename)[0])
    os.makedirs(subdir, exist_ok=True)

    for q in questions:
        fname = f"assessment_{q['assessment_number']}_q{q['question_number']}.png"
        path = os.path.join(subdir, fname)
        with open(path, 'wb') as f:
            f.write(q['image_data'])

    log.info(f"  Saved {len(questions)} images to {subdir}/")


def export_static_site(all_questions, script_dir, unit=None):
    """Export all questions as a static site (images + JSON metadata).

    When unit is provided, writes to unit-scoped paths:
      images -> site/images/{unit}/
      metadata -> site/data/{unit}/questions.json
      image paths in JSON -> images/{unit}/filename.png
    """
    site_dir = os.path.join(script_dir, "site")

    if unit:
        images_dir = os.path.join(site_dir, "images", unit)
        data_dir = os.path.join(site_dir, "data", unit)
        image_prefix = f"images/{unit}"
    else:
        images_dir = os.path.join(site_dir, "images")
        data_dir = os.path.join(site_dir, "data")
        image_prefix = "images"

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    metadata_list = []

    for q in all_questions:
        section_part = f"_{q['section']}" if q['section'] else ""
        image_filename = (
            f"{q['grade']}_{q['unit']}{section_part}"
            f"_a{q['assessment_number']}_q{q['question_number']}.png"
        )
        image_path = os.path.join(images_dir, image_filename)

        with open(image_path, 'wb') as f:
            f.write(q['image_data'])

        metadata_list.append({
            'filename': q['filename'],
            'grade': q['grade'],
            'unit': q['unit'],
            'section': q['section'],
            'assessment_number': q['assessment_number'],
            'question_number': q['question_number'],
            'image': f"{image_prefix}/{image_filename}",
            'image_width': q['image_width'],
            'image_height': q['image_height'],
            'source_pages': q['source_pages'],
        })

    json_path = os.path.join(data_dir, "questions.json")
    with open(json_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)

    unit_label = f" ({unit})" if unit else ""
    log.info(f"\nExported static site{unit_label}:")
    log.info(f"  {len(metadata_list)} question images -> {os.path.relpath(images_dir, script_dir)}/")
    log.info(f"  Metadata -> {os.path.relpath(json_path, script_dir)}")
    log.info(f"  Total image size: {sum(len(q['image_data']) for q in all_questions) / 1024 / 1024:.1f} MB")


def init_db(conn):
    """Create the questions table if it doesn't exist."""
    import psycopg2  # noqa: lazy import
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def store_questions(conn, questions):
    """Insert or update extracted questions in the database."""
    import psycopg2  # noqa: lazy import
    with conn.cursor() as cur:
        for q in questions:
            cur.execute("""
                INSERT INTO questions
                    (filename, grade, unit, section, assessment_number, question_number,
                     image_data, image_width, image_height, source_pages)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (filename, assessment_number, question_number)
                DO UPDATE SET
                    image_data = EXCLUDED.image_data,
                    image_width = EXCLUDED.image_width,
                    image_height = EXCLUDED.image_height,
                    source_pages = EXCLUDED.source_pages,
                    created_at = NOW()
            """, (
                q['filename'], q['grade'], q['unit'], q['section'],
                q['assessment_number'], q['question_number'],
                psycopg2.Binary(q['image_data']),
                q['image_width'], q['image_height'],
                q['source_pages'],
            ))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Extract questions from Illustrative Mathematics PDFs")
    parser.add_argument('--preview', action='store_true',
                        help='Save images locally instead of storing in database')
    parser.add_argument('--all', action='store_true',
                        help='Process all PDFs (default: first only)')
    parser.add_argument('--export-site', action='store_true',
                        help='Export all PDFs as a static site to site/')
    parser.add_argument('--pdf-dir',
                        help='Path to PDF folder (default: IM Assessments)')
    parser.add_argument('--unit',
                        help='Unit ID to process (e.g., G5_U1). Filters PDFs and scopes output.')
    args = parser.parse_args()

    # --export-site implies --all
    if args.export_site:
        args.all = True

    # Resolve PDF directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.pdf_dir:
        # If absolute, use as-is; if relative, resolve from cwd
        pdf_dir = os.path.abspath(args.pdf_dir)
    else:
        pdf_dir = os.path.join(script_dir, PDF_DIR)

    pdf_files = sorted([
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.endswith('.pdf')
    ])

    # Set up logging
    log_file = setup_logging(unit=args.unit)
    log.info(f"Log file: {log_file}")

    # Filter PDFs to only those matching the specified unit
    if args.unit:
        filtered = []
        for pdf_path in pdf_files:
            try:
                meta = parse_filename(pdf_path)
                pdf_unit_id = f"{meta['grade']}_{meta['unit']}"
                if pdf_unit_id == args.unit:
                    filtered.append(pdf_path)
            except ValueError as e:
                log.warning(f"Skipping file (parse error): {os.path.basename(pdf_path)} — {e}")
                continue
        pdf_files = filtered

    if not pdf_files:
        log.error(f"No PDF files found in {pdf_dir}")
        sys.exit(1)

    if not args.all:
        pdf_files = pdf_files[:1]

    log.info(f"Found {len(pdf_files)} PDF(s) to process\n")

    # Connect to database unless in preview or export mode
    conn = None
    if not args.preview and not args.export_site:
        import psycopg2
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            print("ERROR: DATABASE_URL not set. Use --preview or --export-site, or set it in .env")
            sys.exit(1)
        conn = psycopg2.connect(db_url)
        init_db(conn)

    all_questions = []

    for pdf_path in pdf_files:
        basename = os.path.basename(pdf_path)
        log.info(f"Processing: {basename}")

        try:
            metadata = parse_filename(pdf_path)
        except ValueError as e:
            log.error(f"  FAILED to parse filename: {e}")
            continue

        log.info(f"  Grade={metadata['grade']}, Unit={metadata['unit']}, Section={metadata['section']}")

        try:
            questions = extract_questions_from_pdf(pdf_path)
        except Exception as e:
            log.error(f"  FAILED to extract questions: {e}", exc_info=True)
            continue

        log.info(f"  Extracted {len(questions)} questions:")

        for q in questions:
            size_kb = len(q['image_data']) / 1024
            log.info(f"    Assessment {q['assessment_number']}, Q{q['question_number']}: "
                     f"{q['image_width']}x{q['image_height']}px ({size_kb:.0f}KB), "
                     f"pages {q['source_pages']}")

        if args.preview:
            save_preview(questions, basename)
        elif args.export_site:
            all_questions.extend(questions)
        else:
            import psycopg2
            store_questions(conn, questions)
            log.info(f"  Stored {len(questions)} questions in database")

        log.info("")

    if args.export_site:
        export_static_site(all_questions, script_dir, unit=args.unit)

    if conn:
        conn.close()

    log.info("Done!")
    log.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
