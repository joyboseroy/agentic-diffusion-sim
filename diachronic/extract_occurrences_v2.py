"""
extract_occurrences_v2.py
=========================
Same as extract_occurrences.py but:
- Accepts --input flag to specify local graph file
- Falls back to HuggingFace if no local file specified
- Accepts --school_override to force a school label on all edges

Run:
    python3 extract_occurrences_v2.py --input darshana_graph_v2.jsonl \
        --output occurrences_v2.json

    # For Pali suttas with forced school label:
    python3 extract_occurrences_v2.py --input pali_suttas.jsonl \
        --school_override theravada --output occurrences_pali.json
"""

import json, urllib.request, unicodedata, argparse, os
from collections import defaultdict

HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/joyboseroy/darshana-graph"
    "/resolve/main/darshana_graph.jsonl"
)

DEFAULT_TARGETS = [
    "maya", "karma", "dharma", "samsara", "avidya",
    "atman", "brahman", "moksha", "ahimsa", "yoga",
    "sunyata", "nibbana", "nirvana", "anatta", "pratityasamutpada",
    "dhyana", "samadhi", "prajna", "nirodha", "tanha",
    "karuna", "metta", "sila", "upadana", "vedana",
    "moha", "lobha", "dvesha", "cetana", "asrava",
    "jiva", "ajiva", "tapas", "kaivalya",
    "prakriti", "purusha", "pradhana",
    # New Mahayana concepts
    "sunyata", "tathagatagarbha", "bodhicitta", "tathata",
    "dharmadhatu", "upaya", "bodhisattva", "prajna",
    "alayavijnana", "buddha-nature", "skandhas", "rupa",
    "two truths", "svabhava", "madhyamaka", "yogacara",
    "karuna", "metta", "maitri", "paramita",
]

SCHOOL_DATES = {
    "advaita":             (788, 1200),
    "vishishtadvaita":     (1017, 1300),
    "dvaita":              (1238, 1400),
    "achintya_bhedabheda": (1486, 1700),
    "madhyamaka":          (150,  600),
    "yogacara":            (300,  700),
    "theravada":           (-500, -100),
    "vajrayana":           (700, 1200),
    "mahayana":            (100,  900),
    "chan":                (600, 1200),
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

def load_rows(input_path=None, school_override=None):
    rows = []
    if input_path and os.path.exists(input_path):
        print(f"  Loading from local file: {input_path}")
        with open(input_path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line.decode()))
    else:
        print(f"  Downloading from HuggingFace...")
        with urllib.request.urlopen(HUGGINGFACE_URL, timeout=60) as f:
            for line in f:
                rows.append(json.loads(line.decode()))

    if school_override:
        for row in rows:
            row["school"] = school_override
        print(f"  School override: all rows set to '{school_override}'")

    print(f"  Loaded {len(rows):,} rows")
    return rows

def extract_occurrences(rows, target_concepts, min_quote_len=10):
    targets = {normalise(c) for c in target_concepts}
    occurrences = defaultdict(list)
    seen = set()

    for row in rows:
        ca       = normalise(row.get("concept_a", ""))
        cb       = normalise(row.get("concept_b", ""))
        school   = row.get("school", "").strip().lower()
        relation = row.get("relation", "").strip()
        evidence = (row.get("evidence_quote", "")
                    or row.get("evidence", "")
                    or row.get("evidence_quote", "")
                    or "")

        if len(evidence.strip()) < min_quote_len:
            continue

        for target in [ca, cb]:
            if target not in targets:
                continue
            other        = cb if target == ca else ca
            target_is_a  = (target == ca)
            key          = f"{target}|{evidence[:100]}|{relation}|{school}"
            if key in seen:
                continue
            seen.add(key)

            date_range = SCHOOL_DATES.get(school, (0, 1500))
            occurrences[target].append({
                "concept":       target,
                "other_concept": other,
                "relation":      relation,
                "direction":     "A->B" if target_is_a else "B->A",
                "school":        school,
                "date_early":    date_range[0],
                "date_late":     date_range[1],
                "evidence":      evidence.strip(),
                "evidence_len":  len(evidence.strip()),
                "cluster_id":    None,
                "sense_label":   None,
            })

    return occurrences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           type=str, default=None,
        help="Local JSONL graph file (default: download from HuggingFace)")
    parser.add_argument("--output",          type=str, default="occurrences_v2.json")
    parser.add_argument("--concepts",        nargs="+", default=None)
    parser.add_argument("--all_concepts",    action="store_true")
    parser.add_argument("--min_occurrences", type=int, default=5)
    parser.add_argument("--min_quote_len",   type=int, default=10)
    parser.add_argument("--school_override", type=str, default=None,
        help="Force all rows to this school label")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  extract_occurrences_v2.py")
    print("="*65 + "\n")

    rows = load_rows(args.input, args.school_override)

    if args.all_concepts:
        from collections import Counter
        all_c = Counter()
        for row in rows:
            all_c[normalise(row.get("concept_a",""))] += 1
            all_c[normalise(row.get("concept_b",""))] += 1
        targets = [c for c,n in all_c.items()
                   if n >= args.min_occurrences and c]
        print(f"  Extracting {len(targets)} concepts (>={args.min_occurrences} occ)")
    elif args.concepts:
        targets = args.concepts
    else:
        targets = list(set(DEFAULT_TARGETS))
        print(f"  Extracting {len(targets)} default target concepts")

    occurrences = extract_occurrences(rows, targets, args.min_quote_len)

    print(f"\n  Results:")
    print(f"  {'Concept':<30} {'Occ':>6}  Top schools")
    print(f"  {'-'*30} {'-'*6}  {'-'*35}")
    for concept, occs in sorted(occurrences.items(), key=lambda x: -len(x[1])):
        sc = defaultdict(int)
        for o in occs: sc[o["school"]] += 1
        top = ", ".join(f"{s}({n})" for s,n in
                        sorted(sc.items(),key=lambda x:-x[1])[:3])
        print(f"  {concept:<30} {len(occs):>6}  {top}")

    output = {
        "metadata": {
            "total_concepts":    len(occurrences),
            "total_occurrences": sum(len(v) for v in occurrences.values()),
            "targets":           targets,
            "input":             args.input or "HuggingFace",
            "pipeline_step":     "1_occurrence_extraction_v2",
        },
        "concepts": {
            concept: {"count": len(occs), "occurrences": occs}
            for concept, occs in occurrences.items()
        }
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output}")
    print(f"  Next:  python3 cluster_occurrences_v3.py --input {args.output}")

if __name__ == "__main__":
    main()
