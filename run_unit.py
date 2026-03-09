#!/usr/bin/env python3
"""
Single-command orchestrator for the KC extraction pipeline.

Runs the full pipeline for one unit:
  1. Extract question images from PDFs (filtered to the unit)
  2. Extract Knowledge Components via GPT vision API
  3. Auto-deduplicate KCs via GPT
  4. Update units.json manifest

Usage:
    python run_unit.py "path/to/PDFs" --unit G5_U1
    python run_unit.py "path/to/PDFs" --unit G5_U1 --resume
    python run_unit.py "path/to/PDFs" --unit G5_U1 --skip-extract
    python run_unit.py "path/to/PDFs" --unit G5_U1 --skip-dedup
"""

import argparse
import json
import os
import re
import subprocess
import sys


def run_step(description, cmd):
    """Run a subprocess step, streaming output. Exit on failure."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\nERROR: Step failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def grade_label(grade_code):
    """Convert grade code to human-readable label.

    GK -> Kindergarten, G1-G8 -> Grade 1-8,
    G9 -> Algebra 1, G10 -> Geometry, G11 -> Algebra 2
    """
    labels = {
        'GK': 'Kindergarten',
        'G9': 'Algebra 1',
        'G10': 'Geometry',
        'G11': 'Algebra 2',
    }
    if grade_code in labels:
        return labels[grade_code]
    m = re.match(r'G(\d+)', grade_code)
    if m:
        return f"Grade {m.group(1)}"
    return grade_code


def build_units_manifest(script_dir):
    """Scan site/data/*/kcs.json to build site/data/units.json manifest."""
    data_dir = os.path.join(script_dir, "site", "data")
    units = []

    if not os.path.isdir(data_dir):
        return units

    for entry in sorted(os.listdir(data_dir)):
        unit_dir = os.path.join(data_dir, entry)
        kcs_path = os.path.join(unit_dir, "kcs.json")
        questions_path = os.path.join(unit_dir, "questions.json")

        if not os.path.isdir(unit_dir) or not os.path.exists(kcs_path):
            continue

        with open(kcs_path) as f:
            kcs = json.load(f)

        q_count = 0
        if os.path.exists(questions_path):
            with open(questions_path) as f:
                q_count = len(json.load(f))

        # Parse grade and unit number from ID like "G5_U1"
        m = re.match(r'(G\w+)_(U\d+)', entry)
        if m:
            grade_code = m.group(1)
            unit_num = m.group(2)[1:]  # strip 'U' prefix
            label = f"{grade_label(grade_code)} Unit {unit_num}"
        else:
            label = entry

        units.append({
            "id": entry,
            "label": label,
            "kc_count": len(kcs),
            "question_count": q_count,
        })

    manifest_path = os.path.join(data_dir, "units.json")
    with open(manifest_path, 'w') as f:
        json.dump(units, f, indent=2)

    return units


def main():
    parser = argparse.ArgumentParser(
        description="Run the full KC extraction pipeline for one unit"
    )
    parser.add_argument('pdf_dir',
                        help='Path to folder containing assessment PDFs')
    parser.add_argument('--unit', required=True,
                        help='Unit ID (e.g., G5_U1)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip assessments with cached GPT responses')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip step 1 (reuse existing question images)')
    parser.add_argument('--skip-kcs', action='store_true',
                        help='Skip step 2 (reuse existing KC extraction)')
    parser.add_argument('--skip-dedup', action='store_true',
                        help='Skip step 3 (no deduplication)')
    parser.add_argument('--model', default='gpt-5.2-chat-latest',
                        help='OpenAI model for KC extraction and dedup')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.abspath(args.pdf_dir)
    python = sys.executable  # Use same Python interpreter

    if not os.path.isdir(pdf_dir):
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        sys.exit(1)

    pdf_count = len([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
    print(f"KC Extraction Pipeline")
    print(f"  Unit:     {args.unit}")
    print(f"  PDFs:     {pdf_dir} ({pdf_count} files)")
    print(f"  Model:    {args.model}")
    print(f"  Resume:   {args.resume}")

    # Step 1: Extract question images from PDFs
    if not args.skip_extract:
        run_step(
            f"Extract question images for {args.unit}",
            [python, "extract_questions.py", "--export-site",
             "--pdf-dir", pdf_dir, "--unit", args.unit]
        )
    else:
        print(f"\nSkipping question extraction (--skip-extract)")

    # Step 2: Extract KCs via GPT vision API
    if not args.skip_kcs:
        cmd = [python, "extract_kcs.py", "--unit", args.unit, "--model", args.model]
        if args.resume:
            cmd.append("--resume")
        run_step("Extract Knowledge Components via GPT", cmd)
    else:
        print(f"\nSkipping KC extraction (--skip-kcs)")

    # Step 3: Auto-deduplicate KCs
    if not args.skip_dedup:
        run_step(
            "Auto-deduplicate KCs via GPT",
            [python, "dedup_kcs.py", "--unit", args.unit, "--auto",
             "--model", args.model]
        )
    else:
        print(f"\nSkipping deduplication (--skip-dedup)")

    # Step 4: Build units manifest
    print(f"\n{'='*60}")
    print(f"STEP: Build units manifest")
    print(f"{'='*60}")

    units = build_units_manifest(script_dir)
    print(f"\nUnits manifest ({len(units)} units):")
    for u in units:
        print(f"  {u['id']:20s}  {u['kc_count']:4d} KCs  {u['question_count']:4d} questions")

    # Summary
    kcs_path = os.path.join(script_dir, "site", "data", args.unit, "kcs.json")
    questions_path = os.path.join(script_dir, "site", "data", args.unit, "questions.json")

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {args.unit}")
    print(f"{'='*60}")

    if os.path.exists(questions_path):
        with open(questions_path) as f:
            q_count = len(json.load(f))
        print(f"  Questions: {q_count}")

    if os.path.exists(kcs_path):
        with open(kcs_path) as f:
            kc_count = len(json.load(f))
        print(f"  KCs:       {kc_count}")

    print(f"\nDeploy: netlify deploy --prod --dir=site")


if __name__ == "__main__":
    main()
