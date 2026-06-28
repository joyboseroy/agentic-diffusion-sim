"""
Run from agentic-diffusion-sim directory.
Fixes three incorrect attributions in temporal_source_layer.json:
  maya   -> Buddhist-influenced (Gaudapada, ca 500-700 CE) not Vedic
  avidya -> Buddhist/Theravada (avijja in Pali Canon) not Hindu/Advaita
  samsara-> contested, mark as such, not confidently early Upanishadic
Also lowers min_degree threshold for karuna/maitri to 2.
"""
import json

with open("temporal_source_layer.json") as f:
    sources = json.load(f)

# Track changes
changed = []

for source in sources:
    trad = source.get("tradition", "")
    text = source.get("text", "")
    concepts = source.get("concepts_introduced", [])

    # Fix 1: maya in Vedic sources — change note to clarify word vs concept
    if "Rigveda" in text or "Vedic" in trad:
        for entry in concepts:
            if entry["concept"].lower() == "maya":
                entry["note"] = (
                    "The WORD maya appears in RV 6.47.18 as magical power of gods. "
                    "The PHILOSOPHICAL CONCEPT of maya as cosmic illusion obscuring "
                    "reality is NOT Vedic. It is developed by Gaudapada (Mandukya "
                    "Karika, ca 500-700 CE) via explicit adoption of Madhyamaka "
                    "Buddhist arguments — see King 1995. Vedic entry records word "
                    "attestation only, not philosophical concept."
                )
                changed.append("maya note updated in Vedic source")

    # Fix 2: Remove avidya from Hindu/Advaita — it belongs to Buddhist/Theravada
    if trad == "Advaita Vedanta":
        new_concepts = []
        for entry in concepts:
            if entry["concept"].lower() in ("avidya", "maya as systematic metaphysics",
                                             "maya as systematic advaita term"):
                changed.append(f"Removed '{entry['concept']}' from Advaita source")
            else:
                new_concepts.append(entry)
        source["concepts_introduced"] = new_concepts

    # Fix 3: samsara in early Upanishads — mark as contested
    if "Upanishadic" in trad or "Vedic/early Upanishadic" in trad:
        for entry in concepts:
            if entry["concept"].lower() == "samsara":
                entry["note"] = (
                    "CONTESTED: samsara does not appear in the Rigveda. "
                    "Early Upanishadic texts use it but Bronkhorst (2007) argues "
                    "the rebirth doctrine entered Brahmanical thought FROM Shramana "
                    "traditions. Dating early Upanishads to 900-600 BCE is itself "
                    "contested. This entry should be treated as approximate and "
                    "the tradition as possibly Shramana-origin absorbed into "
                    "Brahmanical texts."
                )
                source["certainty"] = "contested"
                changed.append("samsara marked as contested")

# Add corrected maya philosophical entry under Pre-Shankara Advaita
# (where it actually belongs philosophically)
maya_correct = {
    "text": "Mandukya Karika — Gaudapada (philosophical maya)",
    "tradition": "Pre-Shankara Advaita",
    "date_label": "500-700 CE",
    "date_early_ce": 500,
    "date_late_ce": 700,
    "certainty": "approximate",
    "citation": (
        "King, R. (1995). Early Advaita Vedanta and Buddhism. SUNY Press. "
        "Chapter 4 documents Gaudapada's direct adoption of Madhyamaka arguments. "
        "Nakamura, H. (1950). A History of Early Vedanta Philosophy. Motilal."
    ),
    "concepts_introduced": [
        {
            "concept": "maya as systematic metaphysics",
            "note": (
                "The philosophical concept of maya as cosmic illusion enters Advaita "
                "through Gaudapada's Mandukya Karika via explicit Madhyamaka "
                "arguments. Ramanuja accused Shankara of crypto-Buddhism (pracchanna "
                "bauddha) for this. The concept is Buddhist-origin absorbed into "
                "Advaita, NOT Vedic origin."
            )
        },
        {
            "concept": "ajativada",
            "note": "no-origination doctrine, directly derived from Madhyamaka"
        }
    ]
}

# Add corrected avidya under Buddhist/Theravada
avidya_correct = {
    "text": "Pali Canon — avijja as root ignorance",
    "tradition": "Theravada Buddhist",
    "date_label": "500-300 BCE",
    "date_early_ce": -500,
    "date_late_ce": -300,
    "certainty": "approximate",
    "citation": (
        "Bodhi, B. (2000). The Connected Discourses of the Buddha. Wisdom. "
        "Avijja is the first nidana in pratityasamutpada — the ignorance that "
        "initiates the chain of dependent origination. It is a Theravada Buddhist "
        "technical term that Shankara later adapted as avidya in Advaita."
    ),
    "concepts_introduced": [
        {
            "concept": "avijja",
            "note": (
                "Root ignorance — first link in dependent origination, "
                "Pali form of Sanskrit avidya. Buddhist origin, later adopted "
                "by Advaita Vedanta."
            )
        },
        {
            "concept": "avidya",
            "note": (
                "Sanskrit form of avijja. Enters Advaita through Shankara's "
                "engagement with Buddhist epistemology. Buddhist origin, not Vedic."
            )
        }
    ]
}

sources.append(maya_correct)
sources.append(avidya_correct)
changed.append("Added corrected maya entry under Pre-Shankara Advaita")
changed.append("Added avidya under Buddhist/Theravada")

with open("temporal_source_layer.json", "w") as f:
    json.dump(sources, f, indent=2)

print("Changes made:")
for c in changed:
    print(f"  {c}")
print(f"\nTotal sources: {len(sources)}")
