"""
fix_occurrences.py
==================
Post-processes occurrences.json to fix two problems:

1. Merges Pali forms into Sanskrit equivalents
   kamma -> karma, dhamma -> dharma, mokkha -> moksha,
   panna -> prajna, avijja -> avidya, metta -> karuna (compassion)
   
2. Downloads graph once and saves locally as darshana_graph.jsonl
   All subsequent scripts check for local file first.

Run: python3 fix_occurrences.py
Output: occurrences_merged.json  (use this instead of occurrences.json)
"""

import json, urllib.request, os
from collections import defaultdict

HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/joyboseroy/darshana-graph"
    "/resolve/main/darshana_graph.jsonl"
)
LOCAL_GRAPH = "darshana_graph.jsonl"

# Pali -> Sanskrit canonical form
PALI_TO_SANSKRIT = {
    "kamma":          "karma",
    "dhamma":         "dharma",
    "mokkha":         "moksha",
    "panna":          "prajna",
    "avijja":         "avidya",
    "metta":          "karuna",       # loving-kindness -> compassion (both brahmaviharas)
    "bojjhanga":      "bodhyanga",
    "sacca":          "satya",
    "dosa":           "dvesha",
    "jhana":          "dhyana",
    "nibbana":        "nirvana",      # keep nibbana as primary, merge nirvana into it
    "sila":           "sila",         # same
    "lobha":          "lobha",        # same
    "moha":           "moha",         # same
    "tanha":          "tanha",        # keep as primary Pali form
    "vedana":         "vedana",       # keep as primary
    "cetana":         "cetana",       # keep as primary
    "upadana":        "upadana",      # keep as primary
    "citta":          "citta",        # keep as primary
    "asrava":         "asrava",       # keep as primary
}

# Also merge English synonyms into primary terms
ENGLISH_TO_PRIMARY = {
    "ignorance":      "avidya",
    "suffering":      "dukkha",
    "liberation":     "moksha",
    "craving":        "tanha",
    "feeling":        "vedana",
    "cessation":      "nirodha",
    "compassion":     "karuna",
    "impermanence":   "anicca",
}

def download_graph_if_needed():
    if os.path.exists(LOCAL_GRAPH):
        size = os.path.getsize(LOCAL_GRAPH)
        print(f"  Graph already local: {LOCAL_GRAPH} ({size:,} bytes) — skipping download")
        return
    print(f"  Downloading graph to {LOCAL_GRAPH}...")
    urllib.request.urlretrieve(HUGGINGFACE_URL, LOCAL_GRAPH)
    print(f"  Done: {os.path.getsize(LOCAL_GRAPH):,} bytes")

def load_occurrences(path="occurrences.json"):
    with open(path) as f:
        return json.load(f)

def merge_pali_sanskrit(concepts_data):
    """
    For each Pali term that has a Sanskrit equivalent,
    append its occurrences to the Sanskrit term's list.
    Mark each merged occurrence with pali_form=True.
    """
    merges_done = []
    
    for pali, sanskrit in PALI_TO_SANSKRIT.items():
        if pali == sanskrit:
            continue
        if pali not in concepts_data:
            continue
        if sanskrit not in concepts_data:
            # Create the Sanskrit entry if it doesn't exist
            concepts_data[sanskrit] = {"count": 0, "occurrences": []}
        
        pali_occs = concepts_data[pali]["occurrences"]
        if not pali_occs:
            continue
            
        # Tag each occurrence with its Pali origin
        for o in pali_occs:
            o["pali_form"] = pali
            o["merged_into"] = sanskrit
        
        # Append to Sanskrit entry
        before = len(concepts_data[sanskrit]["occurrences"])
        concepts_data[sanskrit]["occurrences"].extend(pali_occs)
        concepts_data[sanskrit]["count"] = len(concepts_data[sanskrit]["occurrences"])
        after = len(concepts_data[sanskrit]["occurrences"])
        
        merges_done.append((pali, sanskrit, len(pali_occs)))
        print(f"  Merged {pali} ({len(pali_occs)} occ) -> {sanskrit} "
              f"({before} -> {after} occ)")
        
        # Keep original entry but mark as merged
        concepts_data[pali]["merged_into"] = sanskrit
        concepts_data[pali]["occurrences"] = []
        concepts_data[pali]["count"] = 0
    
    return concepts_data, merges_done

def patch_extract_script():
    """
    Patch extract_occurrences.py and cluster scripts to use local graph file.
    Prints the one-line change needed.
    """
    print("\n  To make all scripts use local graph, replace this line:")
    print("    with urllib.request.urlopen(HUGGINGFACE_URL, timeout=60) as f:")
    print("  With this in your scripts:")
    print("""
    if os.path.exists('darshana_graph.jsonl'):
        file_handle = open('darshana_graph.jsonl', 'rb')
    else:
        file_handle = urllib.request.urlopen(HUGGINGFACE_URL, timeout=60)
    with file_handle as f:
    """)

def main():
    print("\n" + "="*60)
    print("  fix_occurrences.py")
    print("  1. Download graph once to local file")
    print("  2. Merge Pali forms into Sanskrit equivalents")
    print("="*60 + "\n")
    
    # Step 1: Download once
    download_graph_if_needed()
    
    # Step 2: Load occurrences
    path = "occurrences.json"
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found. Run extract_occurrences.py first.")
        return
    
    data = load_occurrences(path)
    concepts_data = data["concepts"]
    
    print(f"\n  Concepts before merge: {len(concepts_data)}")
    print(f"  Total occurrences: {sum(d['count'] for d in concepts_data.values()):,}")
    
    # Step 3: Merge Pali -> Sanskrit
    print("\n  Merging Pali forms into Sanskrit equivalents:")
    concepts_data, merges = merge_pali_sanskrit(concepts_data)
    
    print(f"\n  Merges done: {len(merges)}")
    print(f"\n  Key concept counts after merge:")
    for c in ["karma","dharma","moksha","prajna","avidya","karuna",
              "dhyana","satya","nirvana","nibbana","tanha","vedana"]:
        if c in concepts_data:
            n = concepts_data[c]["count"]
            pali_note = " (includes Pali)" if any(m[1]==c for m in merges) else ""
            print(f"    {c}: {n}{pali_note}")
    
    # Step 4: Save
    data["concepts"] = concepts_data
    data["metadata"]["pali_sanskrit_merged"] = True
    data["metadata"]["merges"] = [(p,s,n) for p,s,n in merges]
    
    out = "occurrences_merged.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out}")
    print(f"  Now run: python3 cluster_occurrences_v3.py --input {out}")
    
    patch_extract_script()

if __name__ == "__main__":
    main()
