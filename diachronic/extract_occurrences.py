"""
extract_occurrences.py
======================
Step 1 of the diachronic concept graph pipeline.

For each target concept, extracts all occurrences from the darshana-graph
with their evidence quotes, school, relation type, and connected concept.

Output: occurrences.json — one record per edge involving a target concept.

Run from agentic-diffusion-sim directory:
    python3 extract_occurrences.py
    python3 extract_occurrences.py --concepts maya karma dharma
    python3 extract_occurrences.py --all_concepts   # extract everything
    python3 extract_occurrences.py --min_occurrences 10  # only well-attested concepts
"""

import json, urllib.request, unicodedata, argparse, re
from collections import defaultdict

HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/joyboseroy/darshana-graph"
    "/resolve/main/darshana_graph.jsonl"
)

# The ten most contested concepts your teacher raised
# Plus key Buddhist concepts that were underrepresented
DEFAULT_TARGETS = [
    # Contested Vedic/Buddhist/Jain attribution
    "maya", "karma", "dharma", "samsara", "avidya",
    "atman", "brahman", "moksha", "ahimsa", "yoga",
    # Core Buddhist concepts
    "sunyata", "nibbana", "nirvana", "anatta", "pratityasamutpada",
    "dhyana", "samadhi", "prajna", "nirodha", "tanha",
    "karuna", "metta", "sila", "upadana", "vedana",
    "moha", "lobha", "dvesha", "cetana", "asrava",
    # Core Jain concepts
    "jiva", "ajiva", "tapas", "kaivalya",
    # Core Samkhya
    "prakriti", "purusha", "pradhana",
]

# School -> approximate date range CE for context
SCHOOL_DATES = {
    "advaita":             (788, 1200),
    "vishishtadvaita":     (1017, 1300),
    "dvaita":              (1238, 1400),
    "achintya_bhedabheda": (1486, 1700),
    "madhyamaka":          (150,  600),
    "yogacara":            (300,  700),
    "theravada":           (-500, -100),
    "vajrayana":           (700, 1200),
    "zen":                 (700, 1200),
    "jainism":             (-500,  500),
    "samkhya":             (350,   600),
    "yoga":                (400,   600),
    "nyaya":               (200,   600),
    "vaisheshika":         (200,   600),
    "mimamsa":             (200,   600),
    "general":             (0,    1500),
}

def normalise(s):
    s = unicodedata.normalize("NFC", s.strip().lower())
    for a, b in [("ṛ","r"),("ā","a"),("ī","i"),("ū","u"),("ś","s"),
                 ("ṣ","s"),("ṭ","t"),("ḍ","d"),("ṇ","n"),("ñ","n"),
                 ("ḥ","h"),("ṃ","m"),("ṅ","n")]:
        s = s.replace(a, b)
    return s

def load_rows():
    print("  Downloading darshana_graph.jsonl...")
    rows = []
    if os.path.exists("darshana_graph.jsonl"):
        f = open("darshana_graph.jsonl", "rb")
    else:
        f = urllib.request.urlopen(HUGGINGFACE_URL, timeout=60)
    with f as f:
        for line in f:
            rows.append(json.loads(line.decode()))
    print(f"  Loaded {len(rows):,} edges")
    return rows

def extract_occurrences(rows, target_concepts, min_quote_len=10):
    """
    For each row, if concept_a or concept_b matches a target,
    record the full occurrence with context.
    """
    targets = {normalise(c) for c in target_concepts}
    occurrences = defaultdict(list)  # concept -> list of occurrence dicts
    seen = set()  # deduplicate by evidence quote

    for row in rows:
        ca = normalise(row.get("concept_a", ""))
        cb = normalise(row.get("concept_b", ""))
        school = row.get("school", "").strip().lower()
        relation = row.get("relation", "").strip()
        evidence = row.get("evidence_quote", "") or row.get("evidence", "") or ""
        confidence = row.get("confidence", "")

        # Skip rows without meaningful evidence
        if len(evidence.strip()) < min_quote_len:
            continue

        for target in [ca, cb]:
            if target not in targets:
                continue

            # The other concept in this edge
            other = cb if target == ca else ca
            # Which side is the target in the relation
            target_is_a = (target == ca)

            # Deduplicate: same evidence + relation + school
            key = f"{target}|{evidence[:100]}|{relation}|{school}"
            if key in seen:
                continue
            seen.add(key)

            date_range = SCHOOL_DATES.get(school, (0, 1500))

            occ = {
                "concept":        target,
                "other_concept":  other,
                "relation":       relation,
                "direction":      "A->B" if target_is_a else "B->A",
                "school":         school,
                "date_early":     date_range[0],
                "date_late":      date_range[1],
                "evidence":       evidence.strip(),
                "evidence_len":   len(evidence.strip()),
                "confidence":     confidence,
                # Fields for clustering
                "cluster_id":     None,   # filled in by step 2
                "sense_label":    None,   # filled in by step 2
                "embedding":      None,   # filled in by step 2
            }
            occurrences[target].append(occ)

    return occurrences

def print_summary(occurrences, top_n=5):
    print(f"\n{'='*65}")
    print(f"  OCCURRENCE SUMMARY")
    print(f"{'='*65}")
    total = sum(len(v) for v in occurrences.values())
    print(f"  Total occurrences: {total:,}")
    print(f"  Concepts covered:  {len(occurrences)}")
    print()
    print(f"  {'Concept':<25} {'Occurrences':>12}  Top schools")
    print(f"  {'-'*25} {'-'*12}  {'-'*30}")

    for concept, occs in sorted(occurrences.items(),
                                 key=lambda x: -len(x[1])):
        school_counts = defaultdict(int)
        for o in occs:
            school_counts[o["school"]] += 1
        top_schools = ", ".join(
            f"{s}({n})" for s, n in
            sorted(school_counts.items(), key=lambda x: -x[1])[:3]
        )
        print(f"  {concept:<25} {len(occs):>12}  {top_schools}")

    print(f"\n  Sample evidence quotes for 'maya':")
    if "maya" in occurrences:
        for occ in occurrences["maya"][:top_n]:
            ev = occ["evidence"][:120].replace("\n", " ")
            print(f"    [{occ['school']} | {occ['relation']}]")
            print(f"    {ev}...")
            print()

def main():
    parser = argparse.ArgumentParser(
        description="Extract concept occurrences from darshana-graph"
    )
    parser.add_argument("--concepts", nargs="+", default=None,
        help="Specific concepts to extract (default: all 30 targets)")
    parser.add_argument("--all_concepts", action="store_true",
        help="Extract all concepts with enough occurrences")
    parser.add_argument("--min_occurrences", type=int, default=5,
        help="Minimum occurrences to include a concept (with --all_concepts)")
    parser.add_argument("--min_quote_len", type=int, default=10,
        help="Minimum evidence quote length in characters")
    parser.add_argument("--output", type=str, default="occurrences.json",
        help="Output file")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Step 1: Occurrence Extraction")
    print("  darshana-graph -> concept occurrences with evidence")
    print("="*65 + "\n")

    rows = load_rows()

    if args.all_concepts:
        # Collect all concepts and filter by frequency
        all_concepts = defaultdict(int)
        for row in rows:
            all_concepts[normalise(row.get("concept_a",""))] += 1
            all_concepts[normalise(row.get("concept_b",""))] += 1
        targets = [c for c, n in all_concepts.items()
                   if n >= args.min_occurrences and c]
        print(f"  Extracting all {len(targets)} concepts with "
              f">={args.min_occurrences} occurrences")
    elif args.concepts:
        targets = args.concepts
        print(f"  Extracting {len(targets)} specified concepts")
    else:
        targets = DEFAULT_TARGETS
        print(f"  Extracting {len(targets)} default target concepts")

    print(f"  Targets: {', '.join(targets[:10])}"
          + (f" ... and {len(targets)-10} more" if len(targets) > 10 else ""))

    occurrences = extract_occurrences(
        rows, targets, min_quote_len=args.min_quote_len
    )

    print_summary(occurrences)

    # Save
    # Convert to list format for JSON
    output = {
        "metadata": {
            "total_concepts": len(occurrences),
            "total_occurrences": sum(len(v) for v in occurrences.values()),
            "targets": targets,
            "source": "joyboseroy/darshana-graph",
            "pipeline_step": "1_occurrence_extraction",
            "next_step": "python3 cluster_occurrences.py",
        },
        "concepts": {
            concept: {
                "count": len(occs),
                "occurrences": occs
            }
            for concept, occs in occurrences.items()
        }
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {args.output}")
    print(f"  Next step: python3 cluster_occurrences.py --input {args.output}")
    print()

if __name__ == "__main__":
    main()
