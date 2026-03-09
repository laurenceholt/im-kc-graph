#!/usr/bin/env python3
"""
Extract Knowledge Components (KCs) from math assessment questions
using OpenAI's GPT vision API and the KLI framework.

Usage:
    # Process one assessment (for review)
    python extract_kcs.py --assessment G5_U1_SA_a1

    # Process all assessments
    python extract_kcs.py

    # Show plan without making API calls
    python extract_kcs.py --dry-run

    # Skip assessments that already have cached responses
    python extract_kcs.py --resume

    # Process for a specific unit (multi-unit pipeline)
    python extract_kcs.py --unit G5_U1 --resume
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SITE_DIR = "site"
DATA_DIR = os.path.join(SITE_DIR, "data")
QUESTIONS_JSON = os.path.join(DATA_DIR, "questions.json")
KCS_JSON = os.path.join(DATA_DIR, "kcs.json")
RAW_OUTPUT_DIR = "extracted_kcs/raw"
DEFAULT_MODEL = "gpt-5.2-chat-latest"

# --- System prompt (cleaned from KC extractor prompt.rtf) ---
SYSTEM_PROMPT = r"""You are "KC‑Extractor‑Bot," an expert cognitive task analyst trained in the Knowledge‑Learning‑Instruction (KLI) framework.

────────────────────────
SECTION A | KLI QUICK REFERENCE
────────────────────────

The KLI framework links **Knowledge Components (KCs)**—the smallest teachable, testable units of cognition—to the learning processes and instructional methods that build them.

• Each KC pairs a **Condition** ("when this pattern is present…") with a **Response** ("…do or state this").
• Conditions/Responses can be **Constant** (only one value) or **Variable** (many values).
• A **Relationship** may be **Verbal** (explainable) or **Non‑verbal** (intuitive/implicit).
• **Rationale** indicates whether the KC can be logically derived (Yes) or must be memorised/conventional (No).

Classification table:

| cond_type | resp_type | relationship | rationale | → label |
|:--|:--|:--|:--|:--|
| Constant | Constant | Non‑verbal | No | Association |
| Constant | Constant | Verbal | No | Fact |
| Variable | Constant | Non‑verbal | No | Category |
| Variable | Constant | Verbal | No | Concept |
| Variable | Variable | Non‑verbal | No | Production / Skill |
| Variable | Variable | Verbal | No | Rule / Plan |
| Variable | Variable | Verbal | Yes | Principle / Model |

────────────────────────
SECTION B | GLOSSARY OF KC COLUMNS
────────────────────────

• **kc_id** – short, unique snake_case name.
• **title** – a short human-readable title for the KC.
• **description** – a concise description of the KC.
• **condition** – textual pattern(s) that trigger the KC (inputs, givens, goals).
• **response** – action, computation, or declaration produced.
• **cond_type** – "Constant" (only one recognised pattern) or "Variable" (many).
• **resp_type** – "Constant" or "Variable".
• **relationship** – "Verbal" (if a typical student can verbalise it) or "Non‑verbal".
• **rationale** – "Yes" (if the KC is logically derivable / provable); otherwise "No".
• **label** – one of the seven values in the mapping table above.
• **example_refs** – list of question identifiers (e.g. ["Q1", "Q3"]) where the KC appears.
• **example** – a worked example of the KC using markdown with LaTeX (e.g. $\frac{1}{2} + \frac{1}{3}$).
• **notes** – clarifications; append "???" to any low‑confidence field (< 0.8).

────────────────────────
SECTION C | THREE WORKED KC EXAMPLES (Algebra 1)
────────────────────────

| kc_id | condition | response | cond_type | resp_type | relationship | rationale | label | example_refs | notes |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| solve_one_step_add | Equation of the form *x + k = c* | Subtract *k* from both sides | Variable | Variable | Verbal | Yes | Rule | ["Ex 1"] | — |
| graph_y_intercept | Linear eq. in *y = mx + b* | Plot point (0, b) | Variable | Constant | Non‑verbal | No | Category | ["Ex 2"] | Often implicit |
| dist_prop_mult | If term outside parentheses: *k(a + b)* | Compute *ka + kb* | Variable | Variable | Verbal | Yes | Principle | ["Ex 3"] | Distributive Law |

────────────────────────
SECTION D | OUTPUT FORMAT
────────────────────────

Return a single JSON object with a key "kcs" containing an array. Each element is an object with these exact keys:
kc_id, title, description, condition, response, cond_type, resp_type, relationship, rationale, label, example_refs, example, notes

Do NOT include any text outside the JSON object. Do NOT use markdown code fences.

────────────────────────
SECTION E | STEP‑BY‑STEP PROCEDURE
────────────────────────

1. **Decompose** each problem in the PROBLEM_SET into the smallest solution steps (leaf nodes) — note that the problems include solutions but you must generate your own solution steps that arrive at the same solution.
2. For **every** distinct step, draft its Condition (inputs/patterns) and Response (output/action).
3. Decide Constant vs Variable:
   • Constant = only one permissible value/pattern.
   • Variable = multiple permissible values/patterns.
4. Determine Relationship:
   • Can a typical student verbalise the KC? If unsure, list both.
5. Determine Rationale:
   • If the KC's correctness can be *explained or proved* using domain principles → Yes; otherwise No.
6. Use the mapping table in Section A to assign *label*.
7. Merge duplicate KCs (identical Condition+Response semantics).
8. **Alphabetise** kc_id for readability.
9. Add a short description of the KC, add a Title for it, add any relevant notes in a separate column, and add an example of the KC using markdown + latex.
10. Validate that no required column is blank; flag missing info with "???".
11. Output exactly one JSON object with a "kcs" key, no extra commentary.

────────────────────────
SECTION F | CONSTRAINTS
────────────────────────

✔ Be concise yet precise.
✔ The prompt is fully self‑contained.
"""


def question_id(q):
    """Derive canonical question ID from metadata (matches image filename stem)."""
    section_part = f"_{q['section']}" if q.get('section') else ""
    return f"{q['grade']}_{q['unit']}{section_part}_a{q['assessment_number']}_q{q['question_number']}"


def assessment_id_from_key(key):
    """Create human-readable assessment ID from group key tuple."""
    grade, unit, section, assess_num = key
    section_part = f"_{section}" if section else ""
    return f"{grade}_{unit}{section_part}_a{assess_num}"


def group_questions(questions):
    """Group questions into assessments by (grade, unit, section, assessment_number)."""
    groups = defaultdict(list)
    for q in questions:
        key = (q['grade'], q['unit'], q.get('section') or '', q['assessment_number'])
        groups[key].append(q)
    # Sort questions within each group by question_number
    for key in groups:
        groups[key].sort(key=lambda q: q['question_number'])
    return groups


def build_messages(assessment_questions, script_dir):
    """Build the OpenAI chat messages with question images."""
    content = []

    for i, q in enumerate(assessment_questions, 1):
        qid = question_id(q)
        image_path = os.path.join(script_dir, SITE_DIR, q['image'])

        with open(image_path, 'rb') as f:
            image_b64 = base64.standard_b64encode(f.read()).decode('utf-8')

        # Text label before each image
        content.append({
            "type": "text",
            "text": f"Question {i} (ID: {qid}):"
        })

        # The image
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}",
            }
        })

    # Final instruction
    content.append({
        "type": "text",
        "text": (
            "Analyze the above math assessment questions (which include solutions). "
            "Apply the KC extraction procedure from your system instructions. "
            "In the example_refs field, use the Question IDs shown above "
            "(e.g. [\"Q1\", \"Q3\"]). Return ONLY the JSON object."
        )
    })

    return [{"role": "user", "content": content}]


def parse_response(text):
    """Extract KC array from the API response."""
    text = text.strip()

    # Handle markdown code fences if present despite instructions
    if text.startswith("```"):
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse response as JSON: {e}\nResponse (first 500 chars):\n{text[:500]}")

    # Handle {"kcs": [...]} wrapper or bare array
    if isinstance(data, dict):
        if "kcs" in data:
            kcs = data["kcs"]
        else:
            raise ValueError(f"JSON object has no 'kcs' key. Keys: {list(data.keys())}")
    elif isinstance(data, list):
        kcs = data
    else:
        raise ValueError(f"Expected JSON object or array, got {type(data).__name__}")

    if not isinstance(kcs, list):
        raise ValueError(f"Expected 'kcs' to be an array, got {type(kcs).__name__}")

    # Validate required fields
    required_fields = [
        'kc_id', 'title', 'description', 'condition', 'response',
        'cond_type', 'resp_type', 'relationship', 'rationale',
        'label', 'example_refs', 'example', 'notes'
    ]

    for i, kc in enumerate(kcs):
        missing = [f for f in required_fields if f not in kc]
        if missing:
            print(f"  WARNING: KC[{i}] ({kc.get('kc_id', '???')}) missing fields: {missing}")

    return kcs


def map_example_refs(kcs, assessment_questions):
    """Replace Q1/Q2/Ex 1 references with actual question IDs."""
    qid_map = {}
    for i, q in enumerate(assessment_questions, 1):
        qid = question_id(q)
        qid_map[f"Q{i}"] = qid
        qid_map[f"Q {i}"] = qid
        qid_map[str(i)] = qid
        qid_map[f"Ex {i}"] = qid
        qid_map[f"Ex{i}"] = qid
        # Also map the full question ID to itself
        qid_map[qid] = qid

    for kc in kcs:
        refs = kc.get('example_refs', [])
        if isinstance(refs, str):
            refs = [r.strip() for r in refs.split(',')]

        mapped = []
        for ref in refs:
            ref_str = str(ref).strip()
            if ref_str in qid_map:
                mapped.append(qid_map[ref_str])
            else:
                print(f"  WARNING: Unknown ref '{ref_str}' in KC '{kc.get('kc_id', '???')}'")
                mapped.append(ref_str)

        kc['question_ids'] = sorted(set(mapped))
        if 'example_refs' in kc:
            del kc['example_refs']

    return kcs


def merge_kcs(all_kcs):
    """Merge KCs with identical kc_id, combining their question_ids."""
    merged = {}

    for kc in all_kcs:
        kid = kc['kc_id']
        if kid in merged:
            existing = merged[kid]
            existing['question_ids'] = sorted(
                set(existing['question_ids'] + kc.get('question_ids', []))
            )
            # Note differences in key fields
            for field in ['condition', 'response', 'label']:
                if existing.get(field) != kc.get(field):
                    notes = existing.get('notes') or ''
                    if f'[merged: {field} varies]' not in notes:
                        existing['notes'] = (notes + f" [merged: {field} varies]").strip()
        else:
            merged[kid] = kc

    return [merged[k] for k in sorted(merged.keys())]


def process_assessment(key, questions, client, model, script_dir, resume=False, raw_dir=None):
    """Process one assessment group: send images to API and parse KCs."""
    assess_name = assessment_id_from_key(key)
    print(f"\nProcessing: {assess_name} ({len(questions)} questions)")

    # Check for cached response
    if raw_dir is None:
        raw_dir = os.path.join(script_dir, RAW_OUTPUT_DIR)
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"{assess_name}.json")

    if resume and os.path.exists(raw_path):
        print(f"  Resuming from cached response: {raw_path}")
        with open(raw_path, 'r') as f:
            raw_text = f.read()
    else:
        # Build and send API request
        messages = build_messages(questions, script_dir)

        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=8192,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *messages,
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print(f"  ERROR: API call failed: {e}")
            return None

        raw_text = response.choices[0].message.content

        # Log token usage
        usage = response.usage
        print(f"  Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")

        # Cache raw response
        with open(raw_path, 'w') as f:
            f.write(raw_text)

    # Parse and map
    try:
        kcs = parse_response(raw_text)
        kcs = map_example_refs(kcs, questions)
        print(f"  Extracted {len(kcs)} KCs")
        for kc in kcs:
            print(f"    {kc['kc_id']}: {kc.get('title', '???')} [{kc.get('label', '???')}]")
        return kcs
    except ValueError as e:
        print(f"  ERROR: Parse failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract Knowledge Components from math assessment questions"
    )
    parser.add_argument(
        '--assessment',
        help='Process single assessment (e.g., G5_U1_SA_a1)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show assessment groups without making API calls'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Skip assessments that have cached API responses'
    )
    parser.add_argument(
        '--model', default=DEFAULT_MODEL,
        help=f'OpenAI model to use (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--unit',
        help='Unit ID for multi-unit pipeline (e.g., G5_U1)'
    )
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Unit-scoped paths
    if args.unit:
        questions_path = os.path.join(script_dir, SITE_DIR, "data", args.unit, "questions.json")
        kcs_output = os.path.join(script_dir, SITE_DIR, "data", args.unit, "kcs.json")
        raw_dir = os.path.join(script_dir, "extracted_kcs", args.unit, "raw")
    else:
        questions_path = os.path.join(script_dir, QUESTIONS_JSON)
        kcs_output = os.path.join(script_dir, KCS_JSON)
        raw_dir = os.path.join(script_dir, RAW_OUTPUT_DIR)

    # Load questions
    with open(questions_path) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Group by assessment
    groups = group_questions(questions)
    print(f"Found {len(groups)} assessment groups\n")

    # Filter if --assessment specified
    if args.assessment:
        filtered = {k: v for k, v in groups.items()
                    if assessment_id_from_key(k) == args.assessment}
        if not filtered:
            print(f"No assessment matching '{args.assessment}'")
            print(f"Available assessments:")
            for key in sorted(groups.keys()):
                print(f"  {assessment_id_from_key(key)} ({len(groups[key])} questions)")
            sys.exit(1)
        groups = filtered

    # Dry run: show plan and exit
    if args.dry_run:
        print("DRY RUN — would process these assessments:\n")
        for key in sorted(groups.keys()):
            assess_name = assessment_id_from_key(key)
            qs = groups[key]
            q_nums = ", ".join(f"Q{q['question_number']}" for q in qs)
            print(f"  {assess_name}: {len(qs)} questions ({q_nums})")
        print(f"\nTotal: {sum(len(v) for v in groups.values())} questions across {len(groups)} assessments")
        return

    # Initialize OpenAI client
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # Process each assessment
    all_kcs = []
    for key in sorted(groups.keys()):
        kcs = process_assessment(key, groups[key], client, args.model, script_dir,
                                 args.resume, raw_dir=raw_dir)
        if kcs:
            all_kcs.extend(kcs)
        time.sleep(1)  # Rate limit courtesy

    if not all_kcs:
        print("\nNo KCs extracted.")
        return

    # Merge and output
    merged = merge_kcs(all_kcs)

    os.makedirs(os.path.dirname(kcs_output), exist_ok=True)
    with open(kcs_output, 'w') as f:
        json.dump(merged, f, indent=2)

    kcs_rel = os.path.relpath(kcs_output, script_dir)
    print(f"\n{'='*60}")
    print(f"Wrote {len(merged)} unique KCs to {kcs_rel}")
    print(f"Label distribution:")
    label_counts = defaultdict(int)
    for kc in merged:
        label_counts[kc.get('label', '???')] += 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
