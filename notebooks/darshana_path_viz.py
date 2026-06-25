"""
darshana_path_viz.py — STANDALONE

Traces and visualises concept diffusion paths through the darshana-graph.
Shows the actual stepping-stone concepts between Buddhist and Vaishnava traditions.

Usage:
  python3 darshana_path_viz.py                          # all preset journeys
  python3 darshana_path_viz.py --from sunyata --to "krsna consciousness"
  python3 darshana_path_viz.py --animate --from sunyata --to "krsna consciousness"
  python3 darshana_path_viz.py --all_paths              # matrix of all tradition pairs
"""

import json, urllib.request, unicodedata, argparse, os, sys, time
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/joyboseroy/darshana-graph"
    "/resolve/main/darshana_graph.jsonl"
)

DIRTY_SCHOOLS = {"general", "dvaitadvaita", "jain_common", "jain_digambara"}

TRADITION_MAP = {
    "advaita": "Hindu",          "vishishtadvaita": "Hindu",
    "dvaita": "Hindu",           "nyaya": "Hindu",
    "vaisheshika": "Hindu",      "samkhya": "Hindu",
    "yoga": "Hindu",             "mimamsa": "Hindu",
    "purva_mimamsa": "Hindu",    "achintya_bhedabheda": "Vaishnava",
    "madhyamaka": "Buddhist",    "yogacara": "Buddhist",
    "theravada": "Buddhist",     "vajrayana": "Buddhist",
    "zen": "Buddhist",           "jainism": "Jain",
    "carvaka": "Heterodox",      "ajivika": "Heterodox",
}

TRADITION_COLORS = {
    "Hindu":     "#F4A261",
    "Vaishnava": "#9B72CF",
    "Buddhist":  "#2A9D8F",
    "Jain":      "#E63946",
    "Heterodox": "#ADB5BD",
    "Mixed":     "#74c0fc",
}

# Preset journeys — the fascinating cross-tradition paths
PRESET_JOURNEYS = [
    ("sunyata",            "krsna consciousness", "Emptiness → Krishna Consciousness"),
    ("anatta",             "atman",               "No-Self → Eternal Self"),
    ("nirvana",            "moksha",              "Buddhist Liberation → Hindu Liberation"),
    ("pratityasamutpada",  "brahman",             "Dependent Origination → Absolute Reality"),
    ("anicca",             "maya",                "Impermanence → Cosmic Illusion"),
    ("nirvana",            "lila",                "Cessation → Divine Play"),
    ("sunyata",            "brahman",             "Emptiness → Fullness"),
    ("meditation",         "bhakti",              "Mindfulness → Devotion"),
    ("karma",              "grace",               "Karma → Grace"),
]

# ---------------------------------------------------------------------------
# Data loading and graph building
# ---------------------------------------------------------------------------

def normalise(s):
    s = unicodedata.normalize("NFC", s.strip().lower())
    for a, b in [("ṛ","r"),("ā","a"),("ī","i"),("ū","u"),("ś","s"),
                 ("ṣ","s"),("ṭ","t"),("ḍ","d"),("ṇ","n"),("ñ","n"),
                 ("ḥ","h"),("ṃ","m")]:
        s = s.replace(a, b)
    return s

def load_and_build(max_rows=None):
    print("  Downloading darshana_graph.jsonl from HuggingFace...")
    rows = []
    with urllib.request.urlopen(HUGGINGFACE_URL, timeout=30) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line.decode("utf-8")))
    print(f"  Loaded {len(rows):,} edges.")

    rows = [r for r in rows if r.get("school","") not in DIRTY_SCHOOLS]

    G = nx.Graph()
    edge_meta = {}   # (ca,cb) -> list of (relation, school)
    concept_school_counts = defaultdict(lambda: defaultdict(int))

    for row in rows:
        ca = normalise(row["concept_a"])
        cb = normalise(row["concept_b"])
        school = row.get("school","").strip().lower()
        rel   = row.get("relation","").strip()
        if not ca or not cb or ca == cb:
            continue
        concept_school_counts[ca][school] += 1
        concept_school_counts[cb][school] += 1
        key = (min(ca,cb), max(ca,cb))
        if key not in edge_meta:
            edge_meta[key] = []
        edge_meta[key].append((rel, school))
        if G.has_edge(ca, cb):
            G[ca][cb]["weight"] += 1
        else:
            G.add_edge(ca, cb, weight=1)

    # Majority school per concept
    concept_school = {
        c: max(sc, key=sc.get)
        for c, sc in concept_school_counts.items()
    }

    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G, concept_school, edge_meta

# ---------------------------------------------------------------------------
# Path analysis
# ---------------------------------------------------------------------------

def get_tradition(concept, concept_school):
    school = concept_school.get(concept, "")
    return TRADITION_MAP.get(school, "Mixed")

def describe_edge(ca, cb, edge_meta):
    """Return the most common relation on this edge."""
    key = (min(ca,cb), max(ca,cb))
    metas = edge_meta.get(key, [])
    if not metas:
        return "relates-to"
    rel_counts = defaultdict(int)
    for rel, sch in metas:
        rel_counts[rel] += 1
    return max(rel_counts, key=rel_counts.get)

def find_path(G, src, tgt):
    """Find shortest path, return None if not found."""
    src_n = normalise(src)
    tgt_n = normalise(tgt)
    if src_n not in G:
        similar = [n for n in G.nodes() if normalise(src)[:5] in n][:4]
        print(f"  '{src}' not in graph. Similar: {similar}")
        return None
    if tgt_n not in G:
        similar = [n for n in G.nodes() if normalise(tgt)[:5] in n][:4]
        print(f"  '{tgt}' not in graph. Similar: {similar}")
        return None
    try:
        return nx.shortest_path(G, src_n, tgt_n)
    except nx.NetworkXNoPath:
        print(f"  No path between '{src}' and '{tgt}'")
        return None

def print_path(path, concept_school, edge_meta, title=""):
    if title:
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
    print(f"  Path length: {len(path)-1} steps\n")
    for i, concept in enumerate(path):
        tradition = get_tradition(concept, concept_school)
        school    = concept_school.get(concept, "?")
        color_sym = {"Hindu":"🟠","Buddhist":"🟢","Vaishnava":"🟣",
                     "Jain":"🔴","Heterodox":"⚪","Mixed":"🔵"}.get(tradition,"⚫")
        indent = "  " * i
        if i < len(path) - 1:
            rel = describe_edge(concept, path[i+1], edge_meta)
            rel_clean = rel.replace("_"," ").lower()
            print(f"  {indent}{color_sym} {concept:<30} [{school}]")
            print(f"  {indent}   ↓ {rel_clean}")
        else:
            print(f"  {indent}{color_sym} {concept:<30} [{school}]")
    print()

# ---------------------------------------------------------------------------
# Static path visualisation
# ---------------------------------------------------------------------------

def plot_path(path, concept_school, edge_meta, G,
              title="", output_dir=".", filename=None):
    """
    Draw the path as a highlighted subgraph showing:
    - The path nodes large and labelled
    - Their immediate neighbours smaller and faded
    - Edge labels showing the relation type
    - Tradition colours
    """
    # Subgraph: path nodes + one hop neighbours (for context)
    context_nodes = set(path)
    for node in path:
        for nb in list(G.neighbors(node))[:4]:   # max 4 neighbours per path node
            context_nodes.add(nb)
    sub = G.subgraph(context_nodes).copy()

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Layout — force path nodes into a horizontal line
    pos = {}
    path_set = set(path)
    n = len(path)
    for i, node in enumerate(path):
        pos[node] = (i * 3.0, 0.0)

    # Place context nodes above/below
    import random
    random.seed(42)
    for node in sub.nodes():
        if node not in pos:
            # Find which path node this is closest to
            for i, pnode in enumerate(path):
                if G.has_edge(node, pnode):
                    jitter_y = random.choice([-1.5, 1.5]) + random.uniform(-0.3, 0.3)
                    jitter_x = random.uniform(-0.6, 0.6)
                    pos[node] = (i * 3.0 + jitter_x, jitter_y)
                    break
            if node not in pos:
                pos[node] = (random.uniform(0, n*3), random.uniform(-2, 2))

    path_set = set(path)

    # Draw context edges (faded)
    context_edges = [(u,v) for u,v in sub.edges()
                     if u not in path_set or v not in path_set]
    nx.draw_networkx_edges(sub, pos, edgelist=context_edges,
                           ax=ax, alpha=0.12, edge_color="#5c6370", width=0.8)

    # Draw path edges (bright)
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(sub, pos, edgelist=path_edges,
                           ax=ax, alpha=0.9, edge_color="#FFD700",
                           width=3.0, arrows=True,
                           arrowstyle="-|>", arrowsize=20,
                           connectionstyle="arc3,rad=0.1")

    # Draw context nodes (small, faded)
    context_only = [n for n in sub.nodes() if n not in path_set]
    for node in context_only:
        tradition = get_tradition(node, concept_school)
        color = TRADITION_COLORS.get(tradition, "#555")
        nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                               node_color=[color], node_size=120,
                               alpha=0.35, edgecolors="#2d3142", linewidths=0.5)

    # Draw path nodes (large, bright)
    for i, node in enumerate(path):
        tradition = get_tradition(node, concept_school)
        color = TRADITION_COLORS.get(tradition, "#74c0fc")
        is_endpoint = (i == 0 or i == len(path)-1)
        size = 1800 if is_endpoint else 1100
        border = "#FFD700" if is_endpoint else "#ffffff"
        lw = 3.0 if is_endpoint else 1.5
        nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                               node_color=[color], node_size=size,
                               edgecolors=[border], linewidths=lw, alpha=1.0)

    # Labels for path nodes
    path_labels = {n: n.replace(" ", "\n") for n in path}
    nx.draw_networkx_labels(sub, pos, labels=path_labels,
                            ax=ax, font_size=8, font_color="#ffffff",
                            font_weight="bold")

    # Edge relation labels on path edges
    for ca, cb in path_edges:
        rel = describe_edge(ca, cb, edge_meta)
        rel_clean = rel.replace("_"," ").lower()
        mx = (pos[ca][0] + pos[cb][0]) / 2
        my = (pos[ca][1] + pos[cb][1]) / 2 + 0.25
        ax.text(mx, my, rel_clean, ha="center", va="center",
                fontsize=7, color="#FFD700", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1d27",
                          edgecolor="#FFD700", alpha=0.7))

    # Step numbers
    for i, node in enumerate(path):
        x, y = pos[node]
        ax.text(x, y - 0.55, f"step {i}", ha="center", va="center",
                fontsize=6.5, color="#adb5bd", alpha=0.8)

    # Legend
    patches = [mpatches.Patch(color=c, label=t)
               for t, c in TRADITION_COLORS.items()]
    patches.append(mpatches.Patch(color="#FFD700", label="Path / Endpoints"))
    ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#2d3142",
              labelcolor="#f8f9fa", fontsize=9, loc="upper right")

    # Title
    display_title = title or f"{path[0]} → {path[-1]}"
    ax.set_title(f"Concept Path through Darshana Graph\n{display_title}",
                 color="#f8f9fa", fontsize=13, pad=14)
    ax.axis("off")

    plt.tight_layout()
    fname = filename or f"path_{normalise(path[0])[:8]}_{normalise(path[-1])[:8]}.png"
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out

# ---------------------------------------------------------------------------
# Animated path — idea spreading step by step
# ---------------------------------------------------------------------------

def animate_path(path, concept_school, edge_meta, G,
                 title="", output_dir=".", filename=None):
    """
    GIF animation: concept lights up one node at a time,
    showing the idea 'travelling' from tradition to tradition.
    """
    # Build subgraph
    context_nodes = set(path)
    for node in path:
        for nb in list(G.neighbors(node))[:3]:
            context_nodes.add(nb)
    sub = G.subgraph(context_nodes).copy()

    # Layout
    import random
    random.seed(42)
    pos = {}
    n = len(path)
    for i, node in enumerate(path):
        pos[node] = (i * 3.0, 0.0)
    for node in sub.nodes():
        if node not in pos:
            for i, pnode in enumerate(path):
                if G.has_edge(node, pnode):
                    jitter_y = random.choice([-1.4, 1.4]) + random.uniform(-0.2, 0.2)
                    pos[node] = (i * 3.0 + random.uniform(-0.5,0.5), jitter_y)
                    break
            if node not in pos:
                pos[node] = (random.uniform(0, n*3), random.uniform(-1.8, 1.8))

    path_set = set(path)
    context_edges = [(u,v) for u,v in sub.edges()
                     if u not in path_set or v not in path_set]
    path_edges_all = list(zip(path[:-1], path[1:]))

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0f1117")

    frames = len(path) + 2   # one frame per node + 2 pause frames at end

    def draw_frame(frame_idx):
        ax.clear()
        ax.set_facecolor("#0f1117")
        ax.axis("off")

        revealed = min(frame_idx, len(path) - 1)

        # Context edges always faded
        nx.draw_networkx_edges(sub, pos, edgelist=context_edges,
                               ax=ax, alpha=0.10, edge_color="#5c6370", width=0.6)

        # Path edges up to revealed
        revealed_edges = path_edges_all[:revealed]
        if revealed_edges:
            nx.draw_networkx_edges(sub, pos, edgelist=revealed_edges,
                                   ax=ax, alpha=0.9, edge_color="#FFD700",
                                   width=3.0)

        # Context nodes faded
        for node in sub.nodes():
            if node not in path_set:
                tradition = get_tradition(node, concept_school)
                color = TRADITION_COLORS.get(tradition, "#555")
                nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                                       node_color=[color], node_size=100,
                                       alpha=0.25, edgecolors="#2d3142")

        # Path nodes: lit if revealed, dim if not
        for i, node in enumerate(path):
            tradition = get_tradition(node, concept_school)
            color = TRADITION_COLORS.get(tradition, "#74c0fc")
            if i <= revealed:
                # Lit up — idea has arrived
                is_current = (i == revealed)
                size = 1800 if (i==0 or i==len(path)-1) else 1100
                alpha = 1.0
                border = "#FFD700"
                lw = 4.0 if is_current else 1.5
                glow_size = size + 600 if is_current else 0
                if glow_size > 0:
                    nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                                           node_color=["#FFD700"], node_size=glow_size,
                                           alpha=0.25, edgecolors="none")
            else:
                # Not yet reached — dark
                color = "#2d3142"
                size = 800
                alpha = 0.5
                border = "#5c6370"
                lw = 1.0

            nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                                   node_color=[color], node_size=size,
                                   alpha=alpha, edgecolors=[border], linewidths=lw)

        # Labels for all path nodes
        for i, node in enumerate(path):
            x, y = pos[node]
            reached = i <= revealed
            col = "#ffffff" if reached else "#5c6370"
            ax.text(x, y, node.replace(" ","\n"), ha="center", va="center",
                    fontsize=8, color=col, fontweight="bold" if reached else "normal",
                    zorder=10)
            if reached and i < len(path)-1:
                rel = describe_edge(node, path[i+1], edge_meta)
                rel_clean = rel.replace("_"," ").lower()
                mx = (pos[node][0] + pos[path[i+1]][0]) / 2
                my = (pos[node][1] + pos[path[i+1]][1]) / 2 + 0.30
                ax.text(mx, my, rel_clean, ha="center", fontsize=6.5,
                        color="#FFD700", alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#1a1d27",
                                  edgecolor="#FFD700", alpha=0.6))

        # Step indicator
        step_text = (f"Step {revealed} of {len(path)-1}  |  "
                     f"Now at: {path[revealed]}  [{get_tradition(path[revealed], concept_school)}]")
        ax.text(0.5, -0.04, step_text, transform=ax.transAxes,
                ha="center", color="#adb5bd", fontsize=9)

        # Legend
        patches = [mpatches.Patch(color=c, label=t)
                   for t, c in TRADITION_COLORS.items() if c != "#ADB5BD"]
        ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#2d3142",
                  labelcolor="#f8f9fa", fontsize=8, loc="upper right")

        display_title = title or f"{path[0]} → {path[-1]}"
        ax.set_title(f"Concept Journey through Darshana Graph\n{display_title}",
                     color="#f8f9fa", fontsize=12, pad=10)

    ani = FuncAnimation(fig, draw_frame, frames=frames,
                        interval=900, repeat=True, repeat_delay=2000)

    fname = filename or f"anim_{normalise(path[0])[:8]}_{normalise(path[-1])[:8]}.gif"
    out = os.path.join(output_dir, fname)
    ani.save(out, writer=PillowWriter(fps=1))
    plt.close()
    print(f"  Saved animation: {out}")
    return out

# ---------------------------------------------------------------------------
# All-paths matrix
# ---------------------------------------------------------------------------

def print_all_paths_matrix(G, concept_school, edge_meta):
    """Print a matrix showing path lengths between tradition landmark concepts."""
    landmarks = {
        "sunyata":           "Buddhist",
        "nirvana":           "Buddhist",
        "anatta":            "Buddhist",
        "brahman":           "Hindu (Advaita)",
        "atman":             "Hindu (Advaita)",
        "maya":              "Hindu (Advaita)",
        "krsna consciousness":"Vaishnava",
        "bhakti":            "Vaishnava",
        "lila":              "Vaishnava",
        "jiva":              "Jain",
        "ahimsa":            "Jain",
        "karma":             "Cross-tradition",
        "moksha":            "Cross-tradition",
    }

    available = {c: t for c, t in landmarks.items() if c in G}
    missing = [c for c in landmarks if c not in G]
    if missing:
        print(f"\n  Not in graph: {missing}")

    concepts = list(available.keys())
    print(f"\n{'Path length matrix':^80}")
    print(f"{'='*80}")
    header = f"{'':22}" + "".join(f"{c[:10]:>12}" for c in concepts)
    print(header)
    print("-"*80)

    for src in concepts:
        row = f"  {src:<20}"
        for tgt in concepts:
            if src == tgt:
                row += f"{'—':>12}"
            else:
                try:
                    p = nx.shortest_path(G, src, tgt)
                    row += f"{len(p)-1:>12}"
                except:
                    row += f"{'∞':>12}"
        print(row)
    print("="*80)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Darshana Path Visualiser — trace concept journeys across traditions"
    )
    parser.add_argument("--from", dest="src", type=str, default=None)
    parser.add_argument("--to",   dest="tgt", type=str, default=None)
    parser.add_argument("--animate", action="store_true",
                        help="Save animated GIF of the path")
    parser.add_argument("--all_paths", action="store_true",
                        help="Print distance matrix between landmark concepts")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Darshana Path Visualiser")
    print("  Tracing concept journeys across Hindu/Buddhist/Jain traditions")
    print("="*65 + "\n")

    G, concept_school, edge_meta = load_and_build()

    if args.all_paths:
        print_all_paths_matrix(G, concept_school, edge_meta)
        return

    if args.src and args.tgt:
        journeys = [(args.src, args.tgt, f"{args.src} → {args.tgt}")]
    else:
        journeys = PRESET_JOURNEYS

    saved = []
    for src, tgt, title in journeys:
        path = find_path(G, src, tgt)
        if path is None:
            continue
        print_path(path, concept_school, edge_meta, title=title)
        out = plot_path(path, concept_school, edge_meta, G,
                        title=title, output_dir=args.output_dir)
        saved.append(out)
        if args.animate or (args.src and args.tgt):
            animate_path(path, concept_school, edge_meta, G,
                         title=title, output_dir=args.output_dir)

    print(f"\nDone. {len(saved)} path image(s) saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
