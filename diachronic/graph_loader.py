"""
graph_loader.py
===============
Shared utility: load darshana-graph from local file if available,
otherwise download from HuggingFace. All pipeline scripts should
import this instead of directly calling urllib.request.urlopen.

Usage:
    from graph_loader import load_rows
    rows = load_rows()
"""

import json, urllib.request, os

HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/joyboseroy/darshana-graph"
    "/resolve/main/darshana_graph.jsonl"
)
LOCAL_GRAPH = "darshana_graph.jsonl"
DIRTY_SCHOOLS = {"dvaitadvaita", "jain_digambara", "jain_common"}

def load_rows(filter_dirty=True, verbose=True):
    """Load all graph edges, from local file or HuggingFace."""
    if os.path.exists(LOCAL_GRAPH):
        if verbose:
            print(f"  Loading graph from local file: {LOCAL_GRAPH}")
        with open(LOCAL_GRAPH, "rb") as f:
            rows = [json.loads(line) for line in f]
    else:
        if verbose:
            print(f"  Downloading graph from HuggingFace...")
        rows = []
        with urllib.request.urlopen(HUGGINGFACE_URL, timeout=60) as f:
            for line in f:
                rows.append(json.loads(line.decode()))
        # Save locally for next time
        with open(LOCAL_GRAPH, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        if verbose:
            print(f"  Saved to {LOCAL_GRAPH} for future use")
    
    if filter_dirty:
        rows = [r for r in rows if r.get("school","") not in DIRTY_SCHOOLS]
    
    if verbose:
        print(f"  Loaded {len(rows):,} edges")
    return rows

