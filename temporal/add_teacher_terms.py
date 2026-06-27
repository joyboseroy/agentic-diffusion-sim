"""
Run from agentic-diffusion-sim directory.
Adds teacher's suggested terms to temporal_source_layer.json
with correct scholarly citations.
Also adds Pali-Sanskrit mappings for existing entries.
"""
import json

# New source entries to add
new_sources = [
  {
    "text": "Pali Canon — Theravada (practice and psychology terms)",
    "tradition": "Theravada Buddhist",
    "date_label": "500-300 BCE",
    "date_early_ce": -500,
    "date_late_ce": -300,
    "certainty": "approximate",
    "citation": "Bodhi, B. (2000). The Connected Discourses of the Buddha. Wisdom Publications. Norman, K.R. (1983). Pali Literature. Harrassowitz.",
    "concepts_introduced": [
      {"concept": "nirodha",    "note": "cessation — third noble truth, the goal of practice"},
      {"concept": "nivarana",   "note": "hindrances — five mental factors obstructing meditation"},
      {"concept": "asrava",     "note": "taints/outflows — what liberation eliminates (asava in Pali)"},
      {"concept": "tanha",      "note": "craving — second noble truth, origin of suffering"},
      {"concept": "vedana",     "note": "feeling-tone — pleasant, unpleasant, neutral; basis of reactivity"},
      {"concept": "cetana",     "note": "intention — the volitional factor that constitutes karma"},
      {"concept": "upadana",    "note": "clinging — fourth nidana, sustains rebirth"},
      {"concept": "vipassana",  "note": "insight meditation — direct seeing of three characteristics"},
      {"concept": "metta",      "note": "loving-kindness — first brahmavihara (maitri in Sanskrit)"},
      {"concept": "karuna",     "note": "compassion — second brahmavihara"},
      {"concept": "citta",      "note": "mind/consciousness — the knowing faculty"},
      {"concept": "avijja",     "note": "ignorance — root cause of suffering (avidya in Sanskrit)"},
      {"concept": "kamma",      "note": "intentional action — Pali form of karma"},
      {"concept": "dhamma",     "note": "phenomenon/teaching — Pali form of dharma"},
      {"concept": "sila",       "note": "ethical conduct — first element of threefold training"},
      {"concept": "moha",       "note": "delusion — one of three roots of unskillful action"},
      {"concept": "lobha",      "note": "greed — first root of unskillful action"},
      {"concept": "dosa",       "note": "aversion/hatred — second root of unskillful action"},
    ]
  },
  {
    "text": "Pali Canon — Anguttara/Samyutta Nikaya (bojjhanga)",
    "tradition": "Theravada Buddhist",
    "date_label": "500-300 BCE",
    "date_early_ce": -500,
    "date_late_ce": -300,
    "certainty": "approximate",
    "citation": "Bodhi, B. (2012). The Numerical Discourses of the Buddha. Wisdom Publications.",
    "concepts_introduced": [
      {"concept": "bodhyanga",  "note": "factors of enlightenment — seven qualities leading to awakening (bojjhanga in Pali)"},
    ]
  },
  {
    "text": "Dhammapada / early Pali Canon — meditation",
    "tradition": "Theravada Buddhist",
    "date_label": "500-300 BCE",
    "date_early_ce": -500,
    "date_late_ce": -300,
    "certainty": "approximate",
    "citation": "Cousins, L.S. (1973). Buddhist Jhana: its nature and attainment. Religion 3(2).",
    "concepts_introduced": [
      {"concept": "jhana",      "note": "meditative absorption — Pali form of dhyana; the Buddha's core meditation teaching"},
      {"concept": "dhyana",     "note": "meditation — Sanskrit form; systematic cultivation of concentrated awareness"},
    ]
  },
  {
    "text": "Yogacara / Mahayana — obscurations",
    "tradition": "Yogacara Buddhist",
    "date_label": "300-400 CE",
    "date_early_ce": 300,
    "date_late_ce": 400,
    "certainty": "approximate",
    "citation": "Conze, E. (1962). Buddhist Thought in India. Allen and Unwin.",
    "concepts_introduced": [
      {"concept": "avarana",    "note": "obscurations — kleshavaranas (afflictive) and jneyavaranas (cognitive); what liberation removes"},
    ]
  },
]

# Pali-Sanskrit ALIASES to add to exp1_v6_final.py
# (not added to temporal layer — these are lookup aliases)
pali_aliases = {
    "kamma":        "karma",
    "dhamma":       "dharma",
    "nibbana":      "nirvana",    # nirvana already in layer; nibbana = same concept
    "panna":        "prajna",
    "mokkha":       "moksha",
    "avijja":       "avidya",
    "metta":        "maitri",
    "bojjhanga":    "bodhyanga",
    "sacca":        "satya",
    "dosa":         "dvesha",
    "jhana":        "dhyana",
    "khema":        "kshema",
}

# Load and update temporal layer
with open("temporal_source_layer.json") as f:
    existing = json.load(f)

# Check what's already there
existing_concepts = set()
for source in existing:
    for entry in source.get("concepts_introduced", []):
        existing_concepts.add(entry["concept"].strip().lower())

print(f"Existing sources: {len(existing)}")
print(f"Existing concepts: {len(existing_concepts)}")

# Add new sources
merged = existing + new_sources
new_concepts = []
for source in new_sources:
    for entry in source["concepts_introduced"]:
        c = entry["concept"]
        if c.lower() not in existing_concepts:
            new_concepts.append(f"  {c} ({source['tradition']})")

with open("temporal_source_layer.json", "w") as f:
    json.dump(merged, f, indent=2)

print(f"\nMerged: {len(existing)} -> {len(merged)} sources")
print(f"New concepts added ({len(new_concepts)}):")
for c in new_concepts:
    print(c)

print("\nPali-Sanskrit aliases to add to ALIASES dict in exp1_v6_final.py:")
for pali, skt in pali_aliases.items():
    print(f'    "{pali}": "{skt}",')
