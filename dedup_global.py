#!/usr/bin/env python3
"""
Cross-unit KC deduplication.

Detects and merges Knowledge Components that represent the same skill
but appear in different units' kcs.json files.

Usage:
    python dedup_global.py              # Auto-detect and merge cross-unit duplicates
    python dedup_global.py --dry-run    # Show what would be merged without writing
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


# --- GPT prompt for cross-unit dedup ---
GLOBAL_DEDUP_PROMPT = """You are a deduplication specialist for Knowledge Components (KCs) extracted from math assessments across multiple curriculum units.

Below is a list of KC IDs with their titles. The unit in brackets is for context only. The same mathematical skill may appear in multiple units with the same or different kc_ids.

Your task: identify groups of duplicate KCs that should be merged into a single KC. Two KCs are duplicates if they represent the same underlying mathematical skill or concept, even if:
- They have different kc_ids but describe the same skill
- They appear in different grade/unit contexts but test the same competency
- They have slightly different names or phrasings

RULES:
- Each group must have a "canonical" kc_id and a list of "duplicate" kc_ids to merge into it.
- Use ONLY the kc_id (e.g. "pythagorean_theorem_apply"), NOT the unit prefix. Do NOT include unit names in the IDs.
- A kc_id can appear in at most ONE group (either as canonical or as a duplicate).
- Only group KCs that are genuinely the same skill. Do NOT group KCs that are merely related.
- If a kc_id appears in multiple units but is already the same ID, still include it as a group (canonical = that ID, duplicates = empty list is fine, but preferably just skip it since there's nothing to rename).
- The canonical ID should be the clearest, most descriptive name from the group.

Return a JSON object with a single key "merge_groups" containing an object where:
- Each key is the canonical kc_id (just the ID, no unit prefix)
- Each value is an array of duplicate kc_ids to merge into it (just the IDs, no unit prefix)

Example:
{"merge_groups": {"pythagorean_theorem_apply": ["apply_pythagorean_theorem", "pythagorean_theorem_application"]}}

Here are the KCs to analyze:

"""


def load_all_unit_kcs(script_dir):
    """Load KCs from all units, tagging each with its source unit."""
    data_dir = os.path.join(script_dir, "site", "data")
    unit_kcs = {}

    for entry in sorted(os.listdir(data_dir)):
        kcs_path = os.path.join(data_dir, entry, "kcs.json")
        if not os.path.isdir(os.path.join(data_dir, entry)):
            continue
        if not os.path.exists(kcs_path):
            continue

        with open(kcs_path) as f:
            kcs = json.load(f)

        for kc in kcs:
            kc["_unit"] = entry

        unit_kcs[entry] = kcs

    return unit_kcs


def auto_detect_global_merge_groups(unit_kcs, client, model="gpt-5.2-chat-latest"):
    """Use GPT to detect cross-unit duplicate KC groups."""
    # Build KC list with unit context
    lines = []
    all_kc_ids = set()

    for unit_id, kcs in sorted(unit_kcs.items()):
        for kc in kcs:
            lines.append(f"- [{unit_id}] {kc['kc_id']}: {kc['title']}")
            all_kc_ids.add(kc["kc_id"])

    kc_list = "\n".join(lines)
    prompt = GLOBAL_DEDUP_PROMPT + kc_list

    total_kcs = sum(len(kcs) for kcs in unit_kcs.values())
    print(f"  Sending {total_kcs} KCs ({len(all_kc_ids)} distinct IDs) to GPT...")

    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content
    usage = response.usage
    print(f"  Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")

    # Parse response
    text = raw_text.strip()
    if text.startswith("```"):
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    data = json.loads(text)
    raw_groups = data.get("merge_groups", {})

    # Strip any unit prefixes GPT may have added (e.g. "G5_U1.kc_id" -> "kc_id")
    def strip_prefix(kc_id):
        if "." in kc_id:
            parts = kc_id.split(".", 1)
            if re.match(r'G\w+_U\d+', parts[0]):
                return parts[1]
        return kc_id

    groups = {}
    for canonical, dupes in raw_groups.items():
        clean_canonical = strip_prefix(canonical)
        clean_dupes = [strip_prefix(d) for d in dupes]
        # Deduplicate in case stripping prefixes creates duplicates
        seen_dupes = set()
        unique_dupes = []
        for d in clean_dupes:
            if d != clean_canonical and d not in seen_dupes:
                unique_dupes.append(d)
                seen_dupes.add(d)
        if unique_dupes:
            groups[clean_canonical] = unique_dupes

    # Validate: every ID must exist in our KC corpus
    seen = set()
    valid_groups = {}

    for canonical, dupes in groups.items():
        if canonical not in all_kc_ids:
            print(f"  WARNING: canonical '{canonical}' not found, skipping group")
            continue

        valid_dupes = []
        for d in dupes:
            if d not in all_kc_ids:
                print(f"  WARNING: duplicate '{d}' not found, skipping")
                continue
            if d in seen:
                print(f"  WARNING: '{d}' already in another group, skipping")
                continue
            valid_dupes.append(d)
            seen.add(d)

        if valid_dupes:
            if canonical in seen:
                print(f"  WARNING: canonical '{canonical}' already in another group, skipping")
                continue
            seen.add(canonical)
            valid_groups[canonical] = valid_dupes

    return valid_groups


def detect_same_id_duplicates(unit_kcs):
    """Find kc_ids that appear in multiple units (exact ID match)."""
    from collections import Counter
    kc_unit_count = defaultdict(list)
    for unit_id, kcs in unit_kcs.items():
        for kc in kcs:
            kc_unit_count[kc["kc_id"]].append(unit_id)

    # Return kc_ids that exist in 2+ units
    return {kid: units for kid, units in kc_unit_count.items() if len(units) > 1}


def apply_global_merge_groups(unit_kcs, merge_groups):
    """
    Apply cross-unit merge groups.

    For each merge group:
    1. Find all instances of the canonical and duplicate kc_ids across all units
    2. Collect all question_ids from all instances
    3. Pick the instance with the most question_ids as the base
    4. Keep the base in its home unit with all merged question_ids
    5. Remove all other instances from their respective units

    Returns: (updated unit_kcs dict, list of merge reports)
    """
    # Build lookup: kc_id -> [(unit_id, kc_dict), ...]
    kc_index = defaultdict(list)
    for unit_id, kcs in unit_kcs.items():
        for kc in kcs:
            kc_index[kc["kc_id"]].append((unit_id, kc))

    # Track which (unit, kc object) pairs to remove
    remove_set = set()  # set of python id() values for kc dicts to remove
    reports = []

    for canonical_id, dupe_ids in merge_groups.items():
        group_ids = list(dict.fromkeys([canonical_id] + dupe_ids))  # dedupe, preserve order

        # Gather all (unit, kc) instances for every ID in this group
        all_instances = []
        for gid in group_ids:
            for unit_id, kc in kc_index.get(gid, []):
                all_instances.append((unit_id, kc))

        if len(all_instances) <= 1:
            continue

        # Collect all question_ids
        all_qids = set()
        for _, kc in all_instances:
            all_qids.update(kc["question_ids"])

        # Pick the instance with the most questions as the base
        all_instances.sort(key=lambda x: len(x[1]["question_ids"]), reverse=True)
        home_unit, base_kc = all_instances[0]

        # Update the base: use canonical_id and merge all question_ids
        base_kc["kc_id"] = canonical_id
        base_kc["question_ids"] = sorted(all_qids)

        # Mark all other instances for removal
        removed_from = []
        for unit_id, kc in all_instances[1:]:
            remove_set.add(id(kc))
            removed_from.append(f"{unit_id}/{kc['kc_id']}")

        reports.append({
            "canonical": canonical_id,
            "home_unit": home_unit,
            "question_count": len(all_qids),
            "removed": removed_from,
        })

    # Remove merged entries from each unit's KC list
    for unit_id in unit_kcs:
        unit_kcs[unit_id] = [
            kc for kc in unit_kcs[unit_id]
            if id(kc) not in remove_set
        ]
        # Sort alphabetically by title
        unit_kcs[unit_id].sort(key=lambda k: k["title"].lower())

    return unit_kcs, reports


def write_unit_kcs(unit_kcs, script_dir):
    """Write updated kcs.json for each unit, stripping internal tags."""
    data_dir = os.path.join(script_dir, "site", "data")

    for unit_id, kcs in sorted(unit_kcs.items()):
        # Strip internal tags before writing
        for kc in kcs:
            kc.pop("_unit", None)

        kcs_path = os.path.join(data_dir, unit_id, "kcs.json")
        with open(kcs_path, 'w') as f:
            json.dump(kcs, f, indent=2)
        print(f"  {unit_id}: {len(kcs)} KCs")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate KCs across all units"
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Show detected groups without applying')
    parser.add_argument('--model', default='gpt-5.2-chat-latest',
                        help='OpenAI model for dedup analysis')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Load all units
    unit_kcs = load_all_unit_kcs(script_dir)
    total_kcs = sum(len(kcs) for kcs in unit_kcs.values())
    print(f"Loaded {total_kcs} KCs across {len(unit_kcs)} units")
    for unit_id, kcs in sorted(unit_kcs.items()):
        print(f"  {unit_id}: {len(kcs)} KCs")

    # Step 2: Detect cross-unit duplicates via GPT
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # Step 2a: Detect same-ID cross-unit duplicates (no GPT needed)
    same_id_dupes = detect_same_id_duplicates(unit_kcs)
    if same_id_dupes:
        print(f"\n{len(same_id_dupes)} kc_ids appear in multiple units (auto-merge):")
        for kid, units in sorted(same_id_dupes.items()):
            print(f"  {kid}: {units}")

    # Step 2b: Detect different-ID cross-unit duplicates via GPT
    gpt_merge_groups = auto_detect_global_merge_groups(unit_kcs, client, args.model)

    # Cache GPT merge groups for auditability
    cache_dir = os.path.join(script_dir, "extracted_kcs")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "global_merge_groups.json")
    with open(cache_path, 'w') as f:
        json.dump(gpt_merge_groups, f, indent=2)
    print(f"\nCached GPT merge groups to extracted_kcs/global_merge_groups.json")

    # Combine: GPT groups handle ID renames; same-ID dupes are handled
    # automatically by apply_global_merge_groups (it finds all instances
    # of each kc_id across units). We just need to ensure same-ID dupes
    # are included as merge groups too (canonical = the ID, dupes = empty).
    merge_groups = dict(gpt_merge_groups)
    for kid in same_id_dupes:
        if kid not in merge_groups and not any(kid in dupes for dupes in merge_groups.values()):
            merge_groups[kid] = []  # no renames, just consolidate across units

    # Step 3: Show merge plan
    has_renames = {k: v for k, v in merge_groups.items() if v}
    has_same_id = {k: v for k, v in merge_groups.items() if not v}

    if not merge_groups:
        print("\nNo cross-unit duplicates found.")
        return

    if has_renames:
        print(f"\n{len(has_renames)} cross-unit rename groups (different IDs, same skill):")
        for canonical, dupes in sorted(has_renames.items()):
            print(f"  {canonical} <- {dupes}")
    if has_same_id:
        print(f"\n{len(has_same_id)} same-ID cross-unit consolidations:")
        for kid in sorted(has_same_id):
            print(f"  {kid} ({', '.join(same_id_dupes[kid])})")

    if args.dry_run:
        print("\nDry run — no changes applied.")
        return

    # Step 4: Apply merges
    unit_kcs, reports = apply_global_merge_groups(unit_kcs, merge_groups)

    # Step 5: Print report
    new_total = sum(len(kcs) for kcs in unit_kcs.values())
    total_removed = sum(len(r["removed"]) for r in reports)

    print(f"\nMerge report ({len(reports)} groups applied):")
    for r in reports:
        print(f"\n  {r['canonical']} (kept in {r['home_unit']}, {r['question_count']} questions)")
        for removed in r["removed"]:
            print(f"    removed: {removed}")

    print(f"\nBefore: {total_kcs} KC entries")
    print(f"Removed: {total_removed} duplicates")
    print(f"After:  {new_total} KC entries")

    # Step 6: Write back
    print(f"\nWriting updated kcs.json files:")
    write_unit_kcs(unit_kcs, script_dir)

    # Step 7: Rebuild units manifest
    sys.path.insert(0, script_dir)
    from run_unit import build_units_manifest
    units = build_units_manifest(script_dir)

    print(f"\nUpdated units.json ({len(units)} units):")
    for u in units:
        print(f"  {u['id']:10s}  {u['kc_count']:4d} KCs  {u['question_count']:4d} questions")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
