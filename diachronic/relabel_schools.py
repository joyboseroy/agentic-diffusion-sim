"""
relabel_schools.py
==================
Fixes school labels in mahayana_edges_v2.jsonl using exact source strings.
Run from the mahayana/ directory.
"""

import json
from collections import Counter

# Exact source string fragments -> correct school
# Order matters: more specific matches first
SOURCE_TO_SCHOOL = {
    # Theravada
    "PathofPurification":        "theravada",
    "Nyanatiloka":               "theravada",
    "bp601s":                    "theravada",

    # Madhyamaka
    "mdjkr":                     "madhyamaka",   # Dzongsar Khyentse on Madhyamakavatara
    "Candrakirti":               "madhyamaka",
    "ARYADEV":                   "madhyamaka",
    "Full text of [ KAREN LANG": "madhyamaka",
    "mmk verses center":         "madhyamaka",

    # Yogacara
    "Vasubandhu_s-Thirty":       "yogacara",
    "vasubandhu_twenty":         "yogacara",
    "17403806-Abhidharmasamuccaya": "yogacara",  # Asanga
    "Abhidharmasamuccaya":       "yogacara",

    # Chan
    "Blue_Cliff_Record":         "chan",
    "Record of Linji":           "chan",
    "John Blofeld":              "chan",
    "Huang Po":                  "chan",

    # Soto Zen
    "shobogenzo":                "soto_zen",

    # Nyingma
    "beacon_of_certainty":       "nyingma",
    "Beacon of Certainty":       "nyingma",
    "Excerpts+from+Mipham":      "nyingma",
    "patrul-rinpoche":           "nyingma",
    "Kindly bent to ease us VOL":"nyingma",
    "Kindly bent to ease us _":  "nyingma",
    "Kindly_Bent_to_Ease_Us":    "nyingma",
    "Longchenpa-Chos":           "nyingma",
    "Gateway to Knowledge":      "nyingma",

    # Kagyu
    "Gampopa":                   "kagyu",
    "Jewel Ornament":            "kagyu",

    # Pure Land
    "dBET_T2646_Kyogyoshinsho":  "pure_land",   # Shinran
    "dBET_Alpha_T2608_Senchaku": "pure_land",   # Honen
    "Pure_Land_Sutras":          "pure_land",

    # Mahayana (keep as-is — these are genuinely general Mahayana)
    "shantideva":                "mahayana",
    "Vimalakirti":               "mahayana",
    "ksitigarbha":               "mahayana",
    "heartsutra":                "mahayana",
    "Heart":                     "mahayana",
}

def get_school(source):
    for fragment, school in SOURCE_TO_SCHOOL.items():
        if fragment.lower() in source.lower():
            return school
    return "mahayana"  # fallback

rows = []
changed = 0
original_schools = Counter()
new_schools = Counter()

with open("mahayana_edges_v2.jsonl") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            original_schools[r.get("school", "")] += 1
            new_school = get_school(r.get("source", ""))
            if new_school != r.get("school", ""):
                changed += 1
            r["school"] = new_school
            new_schools[new_school] += 1
            rows.append(r)

with open("mahayana_edges_v2_relabelled.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"Total edges: {len(rows)}")
print(f"Labels changed: {changed}")
print()
print("Before:")
for s, n in original_schools.most_common():
    print(f"  {s:<20}: {n:>6}")
print()
print("After:")
for s, n in new_schools.most_common():
    print(f"  {s:<20}: {n:>6}")

