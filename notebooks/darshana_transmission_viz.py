"""
darshana_transmission_viz.py — STANDALONE

Extended path visualiser with two-layer graph:
  Layer 1 — classical darshana concept graph (28k edges, HuggingFace)
  Layer 2 — historical transmission layer (24 figures, 155 hand-curated edges)

The combined graph can route paths through BOTH concept connections
AND historical figures who bridged traditions.

Usage:
  python3 darshana_transmission_viz.py --from sunyata --to "krsna consciousness"
  python3 darshana_transmission_viz.py --from sunyata --to "krsna consciousness" --animate
  python3 darshana_transmission_viz.py --all_paths
  python3 darshana_transmission_viz.py --show_figure Vivekananda
  python3 darshana_transmission_viz.py --list_figures
  python3 darshana_transmission_viz.py --preset             # all 9 preset journeys
"""

import json, urllib.request, unicodedata, argparse, os, random
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# Transmission layer data — embedded so script is truly standalone
# ---------------------------------------------------------------------------

TRANSMISSION_DATA = [
  {"figure":"Ram Mohan Roy","dates":"1772-1833","movement":"Brahmo Samaj",
   "bridges":["Hindu","Christian","Western Liberal"],
   "edges_created":[["brahman","rational inquiry","IS_FOUNDATION_OF"],["brahman","human dignity","IMPLIES"],["vedanta","monotheism","PARALLELS"],["brahman","social reform","MOTIVATES"]]},
  {"figure":"Debendranath Tagore","dates":"1817-1905","movement":"Brahmo Samaj",
   "bridges":["Hindu","Western Liberal"],
   "edges_created":[["brahman","intuition","KNOWN_THROUGH"],["inner experience","brahman","REVEALS"]]},
  {"figure":"Keshab Chandra Sen","dates":"1838-1884","movement":"Brahmo Samaj",
   "bridges":["Hindu","Christian","Islamic","Universal"],
   "edges_created":[["jesus","brahman","POINTS_TOWARD"],["universal religion","brahman","GROUNDED_IN"],["love","bhakti","IS_IDENTICAL_TO"]]},
  {"figure":"Ramakrishna","dates":"1836-1886","movement":"Vedanta / Universal Mysticism",
   "bridges":["Hindu","Buddhist","Islamic","Christian","Tantric"],
   "edges_created":[["nirvikalpa samadhi","fana","IS_IDENTICAL_TO"],["nirvikalpa samadhi","mystical union","IS_IDENTICAL_TO"],["nirvikalpa samadhi","nirvana","IS_IDENTICAL_TO"],["kali","brahman","IS_FORM_OF"],["love of god","jnana","REACHES_SAME_AS"]]},
  {"figure":"Vivekananda","dates":"1863-1902","movement":"Neo-Vedanta",
   "bridges":["Hindu","Western Science","Western Liberal","Universal"],
   "edges_created":[["brahman","unified field","PARALLELS"],["yoga","empirical method","IS_A"],["atman","human potential","GROUNDS"],["karma","causality","IS_A"],["maya","relativity","PARALLELS"],["vedanta","universal religion","PROVIDES"],["consciousness","fundamental reality","IS"]]},
  {"figure":"Helena Blavatsky","dates":"1831-1891","movement":"Theosophy",
   "bridges":["Hindu","Buddhist","Kabbalistic","Hermetic","Gnostic"],
   "edges_created":[["karma","cosmic evolution","DRIVES"],["atman","monad","IS_IDENTICAL_TO"],["brahman","the absolute","IS_IDENTICAL_TO"],["nirvana","devachan","PARALLELS"],["sunyata","akasha","PARALLELS"]]},
  {"figure":"Henry Olcott","dates":"1832-1907","movement":"Theosophy / Buddhist Revival",
   "bridges":["Buddhist","Western Rationalist"],
   "edges_created":[["dhamma","rational ethics","IS_A"],["meditation","empirical method","IS_A"],["nirvana","rational goal","IS_A"]]},
  {"figure":"Annie Besant","dates":"1847-1933","movement":"Theosophy / Indian Independence",
   "bridges":["Hindu","Western Liberal","Buddhist","Universal"],
   "edges_created":[["dharma","self rule","MOTIVATES"],["karma","justice","GROUNDS"],["atman","political freedom","IMPLIES"]]},
  {"figure":"Aurobindo","dates":"1872-1950","movement":"Integral Yoga",
   "bridges":["Hindu","Western Science","Evolutionary Theory"],
   "edges_created":[["brahman","evolution","EXPRESSES_THROUGH"],["matter","consciousness","ASCENDING_TO"],["evolution","spiritual development","IS_A"],["sunyata","supermind","STAGE_BEFORE"],["nirvana","supermind","PARTIAL_REALISATION_OF"],["matter","sacred","IS"],["body","divine","BECOMES"]]},
  {"figure":"Jiddu Krishnamurti","dates":"1895-1986","movement":"Independent",
   "bridges":["Hindu","Buddhist","Western Psychology"],
   "edges_created":[["conditioning","suffering","CAUSES"],["the observer","the observed","IS_IDENTICAL_TO"],["choiceless awareness","nirvana","PARALLELS"],["choiceless awareness","sahaja samadhi","PARALLELS"],["psychological time","maya","PARALLELS"],["conditioning","karma","PARALLELS"]]},
  {"figure":"Aldous Huxley","dates":"1894-1963","movement":"Perennial Philosophy",
   "bridges":["Hindu","Buddhist","Christian Mystical","Western Literary"],
   "edges_created":[["brahman","ground of being","IS_IDENTICAL_TO"],["atman","divine spark","IS_IDENTICAL_TO"],["nirvana","mystical union","IS_IDENTICAL_TO"],["consciousness","fundamental reality","IS"]]},
  {"figure":"Alan Watts","dates":"1915-1973","movement":"Zen / Vedanta populariser",
   "bridges":["Buddhist","Hindu","Western Psychology","Taoism"],
   "edges_created":[["brahman","sunyata","PARALLELS"],["atman","no self","REVEALS"],["tao","brahman","PARALLELS"],["tao","sunyata","PARALLELS"],["self","universe","IS_IDENTICAL_TO"]]},
  {"figure":"Chogyam Trungpa","dates":"1939-1987","movement":"Tibetan Buddhism / Shambhala",
   "bridges":["Buddhist","Western Psychology","Western Art"],
   "edges_created":[["sunyata","basic goodness","IS_FOUNDATION_OF"],["vajrayana","psychotherapy","PARALLELS"],["buddha nature","basic goodness","IS_IDENTICAL_TO"]]},
  {"figure":"Prabhupada","dates":"1896-1977","movement":"ISKCON / Gaudiya Vaishnavism",
   "bridges":["Vaishnava","Western Secular","Global"],
   "edges_created":[["krsna consciousness","consciousness","IS_HIGHEST_FORM_OF"],["bhakti","love","IS"],["krsna","supreme person","IS"],["lila","ultimate reality","IS"],["brahman","krsna","PERSONAL_ASPECT_OF"],["sunyata","incomplete","IS"]]},
  {"figure":"Ramana Maharshi","dates":"1879-1950","movement":"Advaita / Self-Enquiry",
   "bridges":["Hindu","Buddhist","Western Psychology"],
   "edges_created":[["self enquiry","anatta","REACHES"],["who am i","no self","REVEALS"],["pure awareness","nirvana","PARALLELS"],["pure awareness","turiya","IS"],["sahaja samadhi","nirvana","PARALLELS"],["atman","no self","INQUIRY_INTO_REVEALS"]]},
  {"figure":"Nagarjuna","dates":"150-250 CE","movement":"Madhyamaka Buddhism",
   "bridges":["Buddhist","Hindu"],
   "edges_created":[["sunyata","brahman","CRITIQUES"],["sunyata","maya","PARALLELS"],["two truths","brahman","PARALLELS_STRUCTURE_OF"],["madhyama","nirguna brahman","PARALLELS"]]},
  {"figure":"Shankara","dates":"788-820 CE","movement":"Advaita Vedanta",
   "bridges":["Hindu","Buddhist"],
   "edges_created":[["brahman","atman","IS_IDENTICAL_TO"],["nirguna brahman","sunyata","PARALLELS_BUT_DIFFERS"],["jiva","brahman","IS_ULTIMATELY"],["moksha","brahman realisation","IS"]]},
  {"figure":"Ramanuja","dates":"1017-1137 CE","movement":"Vishishtadvaita",
   "bridges":["Hindu Advaita","Vaishnava Bhakti"],
   "edges_created":[["jiva","brahman","BODY_OF"],["ishvara","brahman","IS_WITH_QUALITIES"],["bhakti","moksha","PATH_TO"],["grace","moksha","ENABLES"]]},
  {"figure":"Madhva","dates":"1238-1317 CE","movement":"Dvaita Vedanta",
   "bridges":["Hindu Advaita","Vaishnava"],
   "edges_created":[["vishnu","brahman","IS_SUPREME"],["liberation","individual soul","PRESERVES"],["grace","liberation","REQUIRED_FOR"]]},
  {"figure":"Chaitanya","dates":"1486-1534 CE","movement":"Gaudiya Vaishnavism",
   "bridges":["Vaishnava","Advaita","Dvaita","Bhakti"],
   "edges_created":[["krsna","brahman","IS_PERSONAL_FORM_OF"],["prema","moksha","SURPASSES"],["lila","ultimate reality","IS"],["love","knowledge","SURPASSES"],["ecstasy","brahman realisation","IS"]]},
  {"figure":"Patanjali","dates":"400 CE approx","movement":"Raja Yoga",
   "bridges":["Hindu Samkhya","Buddhist Meditation"],
   "edges_created":[["samadhi","nirvana","PARALLELS"],["dhyana","jhana","IS_IDENTICAL_TO"],["chitta vritti nirodha","nirvana","PARALLELS"],["yoga","buddhist meditation","STRUCTURALLY_IDENTICAL_TO"]]},
  {"figure":"Kabir","dates":"1440-1518 CE","movement":"Sant tradition / Bhakti",
   "bridges":["Hindu","Islamic Sufi","Sikh"],
   "edges_created":[["rama","allah","IS_IDENTICAL_TO"],["brahman","allah","IS_IDENTICAL_TO"],["bhakti","sufi love","IS_IDENTICAL_TO"],["inner experience","scripture","SURPASSES"]]},
  {"figure":"Guru Nanak","dates":"1469-1539 CE","movement":"Sikhism",
   "bridges":["Hindu","Islamic","Sant tradition"],
   "edges_created":[["ik onkar","brahman","PARALLELS"],["ik onkar","allah","PARALLELS"],["nam simran","dhikr","PARALLELS"],["nam simran","japa","PARALLELS"],["seva","karma yoga","IS_A"]]},
  {"figure":"Rumi","dates":"1207-1273 CE","movement":"Sufi Islam",
   "bridges":["Islamic Sufi","Hindu Bhakti","Universal Mysticism"],
   "edges_created":[["fana","nirvana","PARALLELS"],["fana","nirvikalpa samadhi","PARALLELS"],["fana","anatta","PARALLELS"],["ishq","bhakti","IS_IDENTICAL_TO"],["the beloved","brahman","PARALLELS"],["divine love","prema","IS_IDENTICAL_TO"]]}
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/joyboseroy/darshana-graph"
    "/resolve/main/darshana_graph.jsonl"
)
DIRTY_SCHOOLS = {"general","dvaitadvaita","jain_common","jain_digambara"}

TRADITION_MAP = {
    "advaita":"Hindu","vishishtadvaita":"Hindu","dvaita":"Hindu",
    "nyaya":"Hindu","vaisheshika":"Hindu","samkhya":"Hindu",
    "yoga":"Hindu","mimamsa":"Hindu","purva_mimamsa":"Hindu",
    "achintya_bhedabheda":"Vaishnava","madhyamaka":"Buddhist",
    "yogacara":"Buddhist","theravada":"Buddhist","vajrayana":"Buddhist",
    "zen":"Buddhist","jainism":"Jain","carvaka":"Heterodox","ajivika":"Heterodox",
}

NODE_COLORS = {
    "Hindu":     "#F4A261",
    "Vaishnava": "#9B72CF",
    "Buddhist":  "#2A9D8F",
    "Jain":      "#E63946",
    "Heterodox": "#ADB5BD",
    "Sufi":      "#74b9ff",
    "Sikh":      "#55efc4",
    "Universal": "#fd79a8",
    "Figure":    "#FFD700",   # historical figures
    "Mixed":     "#636e72",
}

PRESET_JOURNEYS = [
    ("sunyata",           "krsna consciousness", "Emptiness → Krishna Consciousness"),
    ("anatta",            "atman",               "No-Self → Eternal Self"),
    ("nirvana",           "lila",                "Cessation → Divine Play"),
    ("sunyata",           "brahman",             "Buddhist Emptiness → Hindu Absolute"),
    ("fana",              "nirvana",             "Sufi Annihilation → Buddhist Liberation"),
    ("choiceless awareness","krsna consciousness","Krishnamurti → Prabhupada"),
    ("meditation",        "bhakti",              "Sitting Practice → Devotion"),
    ("pratityasamutpada", "brahman",             "Dependent Origination → Absolute Reality"),
    ("conditioning",      "maya",                "Psychological Conditioning → Cosmic Illusion"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise(s):
    s = unicodedata.normalize("NFC", s.strip().lower())
    for a,b in [("ṛ","r"),("ā","a"),("ī","i"),("ū","u"),("ś","s"),
                ("ṣ","s"),("ṭ","t"),("ḍ","d"),("ṇ","n"),("ñ","n"),
                ("ḥ","h"),("ṃ","m")]:
        s = s.replace(a,b)
    return s

def fig_node(name):
    """Canonical node ID for a historical figure."""
    return f"[{name}]"

# ---------------------------------------------------------------------------
# Build combined graph
# ---------------------------------------------------------------------------

def load_classical(max_rows=None):
    print("  Downloading darshana_graph.jsonl...")
    rows = []
    with urllib.request.urlopen(HUGGINGFACE_URL, timeout=30) as f:
        for i,line in enumerate(f):
            if max_rows and i >= max_rows: break
            rows.append(json.loads(line.decode("utf-8")))
    print(f"  Loaded {len(rows):,} classical edges.")
    rows = [r for r in rows if r.get("school","") not in DIRTY_SCHOOLS]
    return rows

def build_combined_graph(classical_rows):
    G = nx.Graph()
    edge_meta  = {}   # (a,b) -> list of (rel, school_or_figure)
    concept_school = defaultdict(lambda: defaultdict(int))

    # --- Layer 1: classical ---
    for row in classical_rows:
        ca = normalise(row["concept_a"])
        cb = normalise(row["concept_b"])
        sch= row.get("school","").strip().lower()
        rel= row.get("relation","").strip()
        if not ca or not cb or ca==cb: continue
        concept_school[ca][sch] += 1
        concept_school[cb][sch] += 1
        key = (min(ca,cb), max(ca,cb))
        edge_meta.setdefault(key,[]).append((rel, sch, "classical"))
        if G.has_edge(ca,cb):
            G[ca][cb]["weight"] += 1
        else:
            G.add_edge(ca, cb, weight=1, layer="classical")

    # --- Layer 2: transmission figures ---
    figure_info = {}
    for entry in TRANSMISSION_DATA:
        fname = entry["figure"]
        fnode = fig_node(fname)
        figure_info[fnode] = entry
        G.add_node(fnode, node_type="figure", figure=fname,
                   movement=entry["movement"], dates=entry["dates"])

        for (ca_raw, cb_raw, rel) in entry["edges_created"]:
            ca = normalise(ca_raw)
            cb = normalise(cb_raw)
            # Figure → concept edges (bidirectional for pathfinding)
            for concept in [ca, cb]:
                if not G.has_node(concept):
                    G.add_node(concept, node_type="concept")
                if not G.has_edge(fnode, concept):
                    G.add_edge(fnode, concept, weight=3,
                               layer="transmission", rel=rel, figure=fname)
                    key2 = (min(fnode,concept), max(fnode,concept))
                    edge_meta.setdefault(key2,[]).append((rel, fname, "transmission"))

            # Also add the direct concept-concept edge from the transmission layer
            key3 = (min(ca,cb), max(ca,cb))
            edge_meta.setdefault(key3,[]).append((rel, fname, "transmission"))
            if G.has_edge(ca,cb):
                G[ca][cb]["weight"] += 2
            else:
                G.add_edge(ca, cb, weight=2, layer="transmission",
                           rel=rel, figure=fname)

    # Majority school per concept
    cs = {c: max(sc, key=sc.get) for c,sc in concept_school.items() if sc}

    print(f"  Combined graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges "
          f"({len(TRANSMISSION_DATA)} figures overlaid)")
    return G, cs, edge_meta, figure_info

# ---------------------------------------------------------------------------
# Pathfinding
# ---------------------------------------------------------------------------

def get_tradition(node, concept_school):
    if node.startswith("["):
        return "Figure"
    sch = concept_school.get(node,"")
    return TRADITION_MAP.get(sch,"Mixed")

def get_color(node, concept_school):
    t = get_tradition(node, concept_school)
    return NODE_COLORS.get(t, NODE_COLORS["Mixed"])

def edge_label(a, b, edge_meta):
    key = (min(a,b), max(a,b))
    metas = edge_meta.get(key,[])
    if not metas: return "relates-to"
    counts = defaultdict(int)
    for rel,_,_ in metas: counts[rel] += 1
    return max(counts, key=counts.get).replace("_"," ").lower()

def find_path(G, src, tgt):
    sn, tn = normalise(src), normalise(tgt)
    if sn not in G:
        print(f"  '{src}' not found. Similar: {[n for n in G.nodes() if sn[:5] in str(n)][:5]}")
        return None
    if tn not in G:
        print(f"  '{tgt}' not found. Similar: {[n for n in G.nodes() if tn[:5] in str(n)][:5]}")
        return None
    try:
        return nx.shortest_path(G, sn, tn)
    except nx.NetworkXNoPath:
        print(f"  No path found between '{src}' and '{tgt}'")
        return None

def print_path(path, concept_school, edge_meta, figure_info, title=""):
    print(f"\n{'='*68}")
    if title: print(f"  {title}")
    print(f"  Path: {' → '.join(path)}")
    print(f"  Length: {len(path)-1} steps")
    print(f"{'='*68}")
    for i,node in enumerate(path):
        is_figure = node.startswith("[")
        tradition = get_tradition(node, concept_school)
        sym = {"Hindu":"🟠","Buddhist":"🟢","Vaishnava":"🟣","Jain":"🔴",
               "Figure":"⭐","Sufi":"🔵","Sikh":"🟩","Universal":"🩷",
               "Mixed":"⚫","Heterodox":"⚪"}.get(tradition,"⚫")
        indent = "  " * i
        if is_figure:
            fi = figure_info.get(node,{})
            print(f"  {indent}{sym} {node}  [{fi.get('movement','')} | {fi.get('dates','')}]")
            print(f"  {indent}   bridges: {', '.join(fi.get('bridges',[]))}")
        else:
            sch = concept_school.get(node,"?")
            print(f"  {indent}{sym} {node}  [{sch} | {tradition}]")
        if i < len(path)-1:
            rel = edge_label(node, path[i+1], edge_meta)
            print(f"  {indent}   ↓ {rel}")
    print()

# ---------------------------------------------------------------------------
# Static visualisation
# ---------------------------------------------------------------------------

def plot_path(path, concept_school, edge_meta, figure_info, G,
              title="", output_dir=".", filename=None):
    # Context: path nodes + up to 3 neighbours each
    context = set(path)
    for node in path:
        for nb in list(G.neighbors(node))[:3]:
            context.add(nb)
    sub = G.subgraph(context).copy()

    fig, ax = plt.subplots(figsize=(16,9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Layout: path on horizontal spine
    pos = {}
    random.seed(42)
    for i,node in enumerate(path):
        pos[node] = (i*3.5, 0.0)
    for node in sub.nodes():
        if node not in pos:
            for i,pnode in enumerate(path):
                if G.has_edge(node, pnode):
                    jy = random.choice([-1.6,1.6]) + random.uniform(-0.3,0.3)
                    pos[node] = (i*3.5 + random.uniform(-0.5,0.5), jy)
                    break
            if node not in pos:
                pos[node] = (random.uniform(0,len(path)*3.5), random.uniform(-2,2))

    path_set = set(path)

    # Context edges (faded)
    ctx_edges = [(u,v) for u,v in sub.edges()
                 if u not in path_set or v not in path_set]
    nx.draw_networkx_edges(sub, pos, edgelist=ctx_edges, ax=ax,
                           alpha=0.10, edge_color="#444", width=0.7)

    # Path edges (gold)
    path_edges = list(zip(path[:-1],path[1:]))
    nx.draw_networkx_edges(sub, pos, edgelist=path_edges, ax=ax,
                           alpha=0.95, edge_color="#FFD700", width=3.5)

    # Context nodes
    for node in [n for n in sub.nodes() if n not in path_set]:
        color = get_color(node, concept_school)
        is_fig = node.startswith("[")
        nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                               node_color=[color], node_size=150 if is_fig else 80,
                               alpha=0.3, edgecolors="#333", linewidths=0.5)

    # Path nodes
    for i,node in enumerate(path):
        color = get_color(node, concept_school)
        is_fig = node.startswith("[")
        is_end = (i==0 or i==len(path)-1)
        size   = 2200 if is_end else (1800 if is_fig else 1100)
        border = "#FFD700"
        lw     = 3.5 if is_end else (2.5 if is_fig else 1.5)
        # Glow for current
        if is_end:
            nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                                   node_color=["#FFD700"], node_size=size+700,
                                   alpha=0.18, edgecolors="none")
        nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                               node_color=[color], node_size=size,
                               edgecolors=[border], linewidths=lw, alpha=1.0)

    # Labels
    for i,node in enumerate(path):
        x,y = pos[node]
        label = node.replace("[","").replace("]","").replace(" ","\n")
        ax.text(x, y, label, ha="center", va="center",
                fontsize=7.5, color="#ffffff", fontweight="bold", zorder=10)
        ax.text(x, y-0.6, f"step {i}", ha="center", fontsize=6,
                color="#888", zorder=10)

    # Edge relation labels
    for ca,cb in path_edges:
        rel = edge_label(ca, cb, edge_meta)
        mx  = (pos[ca][0]+pos[cb][0])/2
        my  = (pos[ca][1]+pos[cb][1])/2 + 0.28
        ax.text(mx, my, rel, ha="center", fontsize=6.5, color="#FFD700",
                bbox=dict(boxstyle="round,pad=0.15",facecolor="#0d1117",
                          edgecolor="#FFD700",alpha=0.7))

    # Legend
    patches = [mpatches.Patch(color=c,label=t)
               for t,c in NODE_COLORS.items() if t != "Mixed"]
    ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#333",
              labelcolor="#eee", fontsize=8, loc="upper right", ncol=2)

    display_title = title or f"{path[0]} → {path[-1]}"
    ax.set_title(f"Concept Journey — Classical Graph + Historical Transmission Layer\n{display_title}",
                 color="#f8f9fa", fontsize=13, pad=14)
    ax.axis("off")
    plt.tight_layout()

    fname = filename or f"path_{normalise(path[0])[:8]}_{normalise(path[-1])[:8]}.png"
    out   = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_path(path, concept_school, edge_meta, figure_info, G,
                 title="", output_dir=".", filename=None):
    context = set(path)
    for node in path:
        for nb in list(G.neighbors(node))[:3]:
            context.add(nb)
    sub = G.subgraph(context).copy()

    random.seed(42)
    pos = {}
    for i,node in enumerate(path):
        pos[node] = (i*3.5, 0.0)
    for node in sub.nodes():
        if node not in pos:
            for i,pnode in enumerate(path):
                if G.has_edge(node,pnode):
                    jy = random.choice([-1.4,1.4]) + random.uniform(-0.2,0.2)
                    pos[node] = (i*3.5+random.uniform(-0.4,0.4), jy)
                    break
            if node not in pos:
                pos[node]=(random.uniform(0,len(path)*3.5),random.uniform(-1.8,1.8))

    path_set  = set(path)
    ctx_edges = [(u,v) for u,v in sub.edges()
                 if u not in path_set or v not in path_set]
    path_edges= list(zip(path[:-1],path[1:]))

    fig, ax = plt.subplots(figsize=(16,8))
    fig.patch.set_facecolor("#0d1117")
    frames = len(path) + 3

    def draw(fidx):
        ax.clear()
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        revealed = min(fidx, len(path)-1)

        nx.draw_networkx_edges(sub, pos, edgelist=ctx_edges, ax=ax,
                               alpha=0.08, edge_color="#444", width=0.6)
        if revealed > 0:
            nx.draw_networkx_edges(sub, pos,
                                   edgelist=path_edges[:revealed], ax=ax,
                                   alpha=0.95, edge_color="#FFD700", width=3.5)

        # Context nodes
        for node in [n for n in sub.nodes() if n not in path_set]:
            color = get_color(node, concept_school)
            nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                                   node_color=[color], node_size=80,
                                   alpha=0.20, edgecolors="#333")

        # Path nodes
        for i,node in enumerate(path):
            color  = get_color(node, concept_school)
            is_end = (i==0 or i==len(path)-1)
            is_fig = node.startswith("[")
            if i <= revealed:
                is_cur = (i==revealed)
                size   = 2200 if is_end else (1800 if is_fig else 1100)
                border = "#FFD700"
                lw     = 4.0 if is_cur else (2.5 if is_fig else 1.5)
                if is_cur:
                    nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                                           node_color=["#FFD700"],
                                           node_size=size+800,
                                           alpha=0.22, edgecolors="none")
            else:
                color  = "#1e2433"
                size   = 700
                border = "#555"
                lw     = 0.8

            nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                                   node_color=[color], node_size=size,
                                   edgecolors=[border], linewidths=lw, alpha=1.0)

        # Labels
        for i,node in enumerate(path):
            x,y   = pos[node]
            reached = i <= revealed
            color   = "#ffffff" if reached else "#555"
            label   = node.replace("[","").replace("]","").replace(" ","\n")
            fw      = "bold" if reached else "normal"
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=8, color=color, fontweight=fw, zorder=10)
            if reached and i < len(path)-1:
                rel = edge_label(node, path[i+1], edge_meta)
                mx  = (pos[node][0]+pos[path[i+1]][0])/2
                my  = (pos[node][1]+pos[path[i+1]][1])/2 + 0.30
                ax.text(mx, my, rel, ha="center", fontsize=6.5,
                        color="#FFD700", alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="#0d1117",edgecolor="#FFD700",alpha=0.6))

        cur_node = path[revealed]
        is_fig   = cur_node.startswith("[")
        if is_fig:
            fi = figure_info.get(cur_node,{})
            info = f"⭐ {fi.get('movement','')} | {fi.get('dates','')} | bridges: {', '.join(fi.get('bridges',[]))}"
        else:
            trad = get_tradition(cur_node, concept_school)
            info = f"Tradition: {trad}"
        step_txt = (f"Step {revealed}/{len(path)-1}  ·  {cur_node}  ·  {info}")
        ax.text(0.5, -0.03, step_txt, transform=ax.transAxes,
                ha="center", color="#aaa", fontsize=8)

        patches = [mpatches.Patch(color=c,label=t)
                   for t,c in NODE_COLORS.items() if t != "Mixed"]
        ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#333",
                  labelcolor="#eee", fontsize=7, loc="upper right", ncol=2)

        display_title = title or f"{path[0]} → {path[-1]}"
        ax.set_title(f"Concept Journey — Classical + Transmission Layer\n{display_title}",
                     color="#f8f9fa", fontsize=12, pad=10)

    ani = FuncAnimation(fig, draw, frames=frames,
                        interval=1100, repeat=True, repeat_delay=2500)
    fname = filename or f"anim_{normalise(path[0])[:8]}_{normalise(path[-1])[:8]}.gif"
    out   = os.path.join(output_dir, fname)
    ani.save(out, writer=PillowWriter(fps=1))
    plt.close()
    print(f"  Saved animation: {out}")
    return out

# ---------------------------------------------------------------------------
# All-paths matrix
# ---------------------------------------------------------------------------

def all_paths_matrix(G, concept_school, figure_info):
    landmarks = {
        "sunyata":"Buddhist","nirvana":"Buddhist","anatta":"Buddhist",
        "brahman":"Hindu","atman":"Hindu","maya":"Hindu",
        "krsna consciousness":"Vaishnava","bhakti":"Vaishnava","lila":"Vaishnava",
        "fana":"Sufi","ik onkar":"Sikh","karma":"Cross","moksha":"Cross",
        "[Ramakrishna]":"Figure","[Vivekananda]":"Figure",
        "[Alan Watts]":"Figure","[Prabhupada]":"Figure",
    }
    available = {c:t for c,t in landmarks.items() if c in G}
    concepts  = list(available.keys())

    print(f"\n{'Combined Graph — Shortest Path Matrix':^90}")
    print(f"{'(figure nodes shown in [brackets])':^90}")
    print("="*90)
    hdr = f"{'':28}" + "".join(f"{c[:9]:>10}" for c in concepts)
    print(hdr); print("-"*90)

    for src in concepts:
        row = f"  {src:<26}"
        for tgt in concepts:
            if src==tgt:
                row += f"{'—':>10}"
            else:
                try:
                    p = nx.shortest_path(G, src, tgt)
                    row += f"{len(p)-1:>10}"
                except:
                    row += f"{'∞':>10}"
        print(row)
    print("="*90)

# ---------------------------------------------------------------------------
# Show figure
# ---------------------------------------------------------------------------

def show_figure(name, G, concept_school, edge_meta, figure_info, output_dir="."):
    fnode = fig_node(name)
    if fnode not in G:
        # fuzzy match
        matches = [k for k in figure_info if name.lower() in k.lower()]
        if matches:
            fnode = matches[0]
            name  = fnode.replace("[","").replace("]","")
        else:
            print(f"Figure '{name}' not found. Use --list_figures.")
            return

    fi = figure_info.get(fnode,{})
    print(f"\n{'='*65}")
    print(f"  {fnode}  [{fi.get('movement','')} | {fi.get('dates','')}]")
    print(f"  Bridges: {', '.join(fi.get('bridges',[]))}")
    print(f"  Key move: {fi.get('key_move','')}")
    print(f"\n  Edges created ({len(fi.get('edges_created',[]))} total):")
    for ca,cb,rel in fi.get("edges_created",[]):
        print(f"    {ca}  --[{rel.replace('_',' ').lower()}]-->  {cb}")
    print(f"{'='*65}")

    # Draw ego network of this figure
    neighbours = list(G.neighbors(fnode))
    sub_nodes  = [fnode] + neighbours[:20]
    sub        = G.subgraph(sub_nodes).copy()

    fig_plt, ax = plt.subplots(figsize=(12,8))
    fig_plt.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    pos = nx.spring_layout(sub, seed=42, k=1.5)

    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.4,
                           edge_color="#FFD700", width=1.5)
    nx.draw_networkx_nodes(sub, pos, nodelist=[fnode], ax=ax,
                           node_color=["#FFD700"], node_size=2500,
                           edgecolors="#ffffff", linewidths=2.5)
    for node in neighbours[:20]:
        color = get_color(node, concept_school)
        nx.draw_networkx_nodes(sub, pos, nodelist=[node], ax=ax,
                               node_color=[color], node_size=900,
                               edgecolors="#555", linewidths=1.0)

    labels = {n: n.replace("[","").replace("]","").replace(" ","\n")
              for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax,
                            font_size=7, font_color="#ffffff")

    patches = [mpatches.Patch(color=c,label=t)
               for t,c in NODE_COLORS.items() if t!="Mixed"]
    ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#333",
              labelcolor="#eee", fontsize=8, loc="lower left")
    ax.set_title(f"Transmission Bridges of {name}\n"
                 f"{fi.get('movement','')} | {fi.get('dates','')}",
                 color="#f8f9fa", fontsize=13, pad=12)
    ax.axis("off")
    plt.tight_layout()
    out = os.path.join(output_dir, f"figure_{normalise(name)[:15]}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved ego-network: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Darshana Transmission Visualiser — classical graph + historical figures"
    )
    parser.add_argument("--from",  dest="src", default=None)
    parser.add_argument("--to",    dest="tgt", default=None)
    parser.add_argument("--animate",      action="store_true")
    parser.add_argument("--all_paths",    action="store_true")
    parser.add_argument("--preset",       action="store_true",
                        help="Run all 9 preset journeys")
    parser.add_argument("--show_figure",  type=str, default=None,
                        help="Show ego-network of a historical figure")
    parser.add_argument("--list_figures", action="store_true")
    parser.add_argument("--output_dir",   default=".")
    args = parser.parse_args()

    print("\n" + "="*68)
    print("  Darshana Transmission Visualiser")
    print("  Classical concept graph + 24 historical figures overlaid")
    print("="*68 + "\n")

    if args.list_figures:
        print("  Available figures:")
        for e in TRANSMISSION_DATA:
            print(f"    {e['figure']:<30} [{e['movement']} | {e['dates']}]")
        return

    classical_rows = load_classical()
    G, concept_school, edge_meta, figure_info = build_combined_graph(classical_rows)

    if args.all_paths:
        all_paths_matrix(G, concept_school, figure_info)
        return

    if args.show_figure:
        show_figure(args.show_figure, G, concept_school, edge_meta,
                    figure_info, args.output_dir)
        return

    journeys = ([(args.src, args.tgt, f"{args.src} → {args.tgt}")]
                if args.src and args.tgt else PRESET_JOURNEYS
                if args.preset else PRESET_JOURNEYS[:1])

    for src, tgt, title in journeys:
        path = find_path(G, src, tgt)
        if path is None:
            continue
        print_path(path, concept_school, edge_meta, figure_info, title=title)
        plot_path(path, concept_school, edge_meta, figure_info, G,
                  title=title, output_dir=args.output_dir)
        if args.animate or (args.src and args.tgt):
            animate_path(path, concept_school, edge_meta, figure_info, G,
                         title=title, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
