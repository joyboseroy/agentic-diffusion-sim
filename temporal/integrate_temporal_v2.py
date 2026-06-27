"""
integrate_temporal_v2.py — STANDALONE

Version 2 of the temporal graph integration.

Key improvement over v1:
  Every concept in the graph now gets a date range, not just the 24
  concepts with explicit citations in temporal_source_layer.json.

How the date assignment works (three tiers):

  TIER 1 — Explicit citation (24 concepts, 98 datings):
    From temporal_source_layer.json. Earliest dated scholarly source.
    These are the most reliable dates.

  TIER 2 — School founding date (majority of concepts):
    If a concept has no explicit temporal citation, use the founding
    date of the school that mentions it most in the corpus.
    This is a lower bound: the concept cannot predate its school,
    but it may be older. Flagged as "school_proxy" certainty.

  TIER 3 — Unknown (concepts in unrecognised schools):
    A small number of concepts belong to schools not in our
    founding-date table. These get date 9999 (excluded from eras).

Result: era snapshots now show thousands of concepts, not 17.

Usage:
  python3 integrate_temporal_v2.py                    # full run
  python3 integrate_temporal_v2.py --era -400         # 400 BCE snapshot
  python3 integrate_temporal_v2.py --era 200          # 200 CE snapshot
  python3 integrate_temporal_v2.py --era 800          # 800 CE snapshot
  python3 integrate_temporal_v2.py --compare_rankings # static vs temporal
  python3 integrate_temporal_v2.py --era_betweenness  # all 5 key eras
  python3 integrate_temporal_v2.py --save_graph       # save graphml
"""

import json, urllib.request, unicodedata, argparse, os
from collections import defaultdict
import networkx as nx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/joyboseroy/darshana-graph"
    "/resolve/main/darshana_graph.jsonl"
)
DIRTY_SCHOOLS = {"general","dvaitadvaita","jain_common","jain_digambara"}

# School founding dates — these are the Tier 2 proxy dates
# A concept tagged to this school cannot predate this year
SCHOOL_FOUNDING = {
    # Vedic — pre-philosophical
    "vedic":              -1500,
    # Upanishadic period
    "upanishadic":        -900,
    # Shramana traditions
    "jainism":            -500,
    "theravada":          -300,
    # Early Hindu philosophical schools
    "nyaya":               200,
    "vaisheshika":         200,
    "mimamsa":             200,
    "purva_mimamsa":       200,
    # Buddhist developments
    "madhyamaka":          150,
    "mahayana":            100,
    "yogacara":            300,
    "vajrayana":           700,
    "zen":                 700,
    "chan":                700,
    # Hindu systematic philosophy
    "samkhya":             350,
    "yoga":                400,
    # Vedanta schools — all post-Shankara or contemporary
    "advaita":             788,
    "vishishtadvaita":    1017,
    "dvaita":             1238,
    "achintya_bhedabheda":1486,
    # Sant / Sikh / Sufi
    "sant":               1400,
    "sikhism":            1469,
    "sufi":               900,
    # Modern
    "neo_vedanta":        1863,
    "theosophy":          1875,
}

TRADITION_DISPLAY = {
    "advaita":             "Hindu/Advaita (est. 788 CE)",
    "vishishtadvaita":     "Hindu/Vaishnava (est. 1017 CE)",
    "dvaita":              "Hindu/Vaishnava (est. 1238 CE)",
    "achintya_bhedabheda": "Hindu/Vaishnava (est. 1486 CE)",
    "nyaya":               "Hindu/Nyaya (est. 200 CE)",
    "vaisheshika":         "Hindu/Vaisheshika (est. 200 CE)",
    "samkhya":             "Hindu/Samkhya (est. 350 CE)",
    "yoga":                "Hindu/Yoga (est. 400 CE)",
    "mimamsa":             "Hindu/Mimamsa (est. 200 CE)",
    "purva_mimamsa":       "Hindu/Mimamsa (est. 200 CE)",
    "madhyamaka":          "Buddhist/Madhyamaka (est. 150 CE)",
    "yogacara":            "Buddhist/Yogacara (est. 300 CE)",
    "theravada":           "Buddhist/Theravada (est. -300 CE)",
    "vajrayana":           "Buddhist/Vajrayana (est. 700 CE)",
    "zen":                 "Buddhist/Zen (est. 700 CE)",
    "jainism":             "Jain (est. -500 CE)",
    "carvaka":             "Heterodox",
    "ajivika":             "Heterodox",
}

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

def era_label(year):
    if year < 0:
        return f"{abs(year)} BCE"
    return f"{year} CE"

# ---------------------------------------------------------------------------
# Load temporal layer (Tier 1)
# ---------------------------------------------------------------------------

def load_temporal_layer(path="temporal_source_layer.json"):
    with open(path) as f:
        sources = json.load(f)
    timeline = {}
    for source in sources:
        date = source["date_early_ce"]
        for entry in source["concepts_introduced"]:
            concept = normalise(entry["concept"])
            if concept not in timeline or date < timeline[concept]["date"]:
                timeline[concept] = {
                    "date":       date,
                    "date_late":  source["date_late_ce"],
                    "date_label": source["date_label"],
                    "tradition":  source["tradition"],
                    "text":       source["text"],
                    "certainty":  source["certainty"],   # explicit
                    "citation":   source["citation"],
                    "note":       entry["note"],
                    "tier":       1,
                }
    return timeline

# ---------------------------------------------------------------------------
# Load darshana-graph
# ---------------------------------------------------------------------------

def load_darshana(max_rows=None):
    print("  Downloading darshana_graph.jsonl from HuggingFace...")
    rows = []
    with urllib.request.urlopen(HUGGINGFACE_URL, timeout=30) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line.decode("utf-8")))
    print(f"  Loaded {len(rows):,} edges.")
    return [r for r in rows if r.get("school","") not in DIRTY_SCHOOLS]

# ---------------------------------------------------------------------------
# Build enriched graph with three-tier dating
# ---------------------------------------------------------------------------

def build_enriched_graph(rows, timeline):
    G = nx.Graph()
    corpus_school_counts  = defaultdict(lambda: defaultdict(int))
    edge_counts           = defaultdict(int)

    for row in rows:
        ca  = normalise(row["concept_a"])
        cb  = normalise(row["concept_b"])
        sch = row.get("school","").strip().lower()
        if not ca or not cb or ca == cb:
            continue
        corpus_school_counts[ca][sch] += 1
        corpus_school_counts[cb][sch] += 1
        key = (min(ca,cb), max(ca,cb))
        edge_counts[key] += 1

    tier_counts = {1:0, 2:0, 3:0}

    for concept, school_counts in corpus_school_counts.items():
        static_school = max(school_counts, key=school_counts.get)

        # --- Date assignment ---
        if concept in timeline:
            # Tier 1: explicit scholarly citation
            t = timeline[concept]
            assigned_date      = t["date"]
            assigned_tradition = t["tradition"]
            assigned_text      = t["text"]
            assigned_certainty = t["certainty"]
            assigned_tier      = 1
            tier_counts[1] += 1

        elif static_school in SCHOOL_FOUNDING:
            # Tier 2: school proxy date
            assigned_date      = SCHOOL_FOUNDING[static_school]
            assigned_tradition = TRADITION_DISPLAY.get(
                                   static_school, static_school)
            assigned_text      = f"proxy: {static_school} school founding"
            assigned_certainty = "school_proxy"
            assigned_tier      = 2
            tier_counts[2] += 1

        else:
            # Tier 3: unknown
            assigned_date      = 9999
            assigned_tradition = "Unknown"
            assigned_text      = ""
            assigned_certainty = "unknown"
            assigned_tier      = 3
            tier_counts[3] += 1

        # Mismatch: does temporal date predate static school?
        static_school_date = SCHOOL_FOUNDING.get(static_school, 9999)
        mismatch     = (assigned_date < static_school_date - 100
                        and assigned_tier == 1)
        mismatch_gap = max(0, static_school_date - assigned_date) if mismatch else 0

        G.add_node(concept,
            # Static
            static_school     = static_school,
            static_tradition  = TRADITION_DISPLAY.get(
                                  static_school, static_school),
            static_date       = static_school_date,
            corpus_count      = sum(school_counts.values()),
            # Temporal
            temporal_date     = assigned_date,
            temporal_tradition= assigned_tradition,
            temporal_text     = assigned_text,
            temporal_certainty= assigned_certainty,
            temporal_tier     = assigned_tier,
            # Mismatch
            mismatch          = mismatch,
            mismatch_gap      = mismatch_gap,
        )

    for (ca,cb), weight in edge_counts.items():
        if weight >= 1:
            G.add_edge(ca, cb, weight=weight)

    print(f"\n  Date assignment coverage:")
    print(f"    Tier 1 (explicit citation):  {tier_counts[1]:>5} concepts")
    print(f"    Tier 2 (school proxy date):  {tier_counts[2]:>5} concepts")
    print(f"    Tier 3 (unknown, excluded):  {tier_counts[3]:>5} concepts")

    return G

# ---------------------------------------------------------------------------
# Era subgraph
# ---------------------------------------------------------------------------

def build_era_subgraph(G, era_ce):
    """Return subgraph of concepts whose temporal_date <= era_ce."""
    valid = [n for n in G.nodes()
             if G.nodes[n].get("temporal_date", 9999) <= era_ce]
    return G.subgraph(valid).copy()

# ---------------------------------------------------------------------------
# Era snapshot with betweenness
# ---------------------------------------------------------------------------

def era_snapshot(G, era_ce, top_n=15, verbose=True):
    sub = build_era_subgraph(G, era_ce)
    label = era_label(era_ce)

    print(f"\n{'='*72}")
    print(f"  ERA SNAPSHOT: {label}")
    print(f"  Network: {sub.number_of_nodes():,} concepts, "
          f"{sub.number_of_edges():,} edges")
    print(f"{'='*72}")

    if sub.number_of_nodes() < 5:
        print("  Too few nodes for meaningful analysis.")
        return

    print(f"  Computing betweenness centrality at {label}...")
    bc = nx.betweenness_centrality(sub, normalized=True)
    top = sorted(bc, key=bc.get, reverse=True)[:top_n]

    print(f"\n  Top {top_n} concepts by betweenness at {label}:")
    print(f"  {'Rank':<5} {'Concept':<28} {'Betw':>8}  "
          f"{'Temporal tradition':<35} {'Tier'}")
    print(f"  {'-'*5} {'-'*28} {'-'*8}  {'-'*35} {'-'*4}")

    for i, concept in enumerate(top, 1):
        b  = bc[concept]
        at = sub.nodes[concept]
        tt = at.get("temporal_tradition","?")[:33]
        tier = at.get("temporal_tier","?")
        print(f"  {i:<5} {concept:<28} {b:>8.4f}  {tt:<35} T{tier}")

    # Tradition breakdown
    tradition_counts = defaultdict(int)
    for node in sub.nodes():
        trad = sub.nodes[node].get("temporal_tradition","Unknown")
        # Simplify for display
        if "Vedic" in trad:
            key = "Vedic / early Upanishadic"
        elif "Buddhist/Theravada" in trad:
            key = "Buddhist / Theravada"
        elif "Buddhist/Madhyamaka" in trad:
            key = "Buddhist / Madhyamaka"
        elif "Buddhist/Yogacara" in trad:
            key = "Buddhist / Yogacara"
        elif "Buddhist/Vajrayana" in trad:
            key = "Buddhist / Vajrayana"
        elif "Buddhist/Zen" in trad:
            key = "Buddhist / Zen-Chan"
        elif "Jain" in trad:
            key = "Jain"
        elif "Advaita" in trad:
            key = "Hindu / Advaita"
        elif "Vaishnava" in trad:
            key = "Hindu / Vaishnava"
        elif "Samkhya" in trad:
            key = "Hindu / Samkhya"
        elif "Yoga" in trad and "Buddhist" not in trad:
            key = "Hindu / Yoga"
        elif "Nyaya" in trad or "Vaisheshika" in trad:
            key = "Hindu / Nyaya-Vaisheshika"
        elif "Mimamsa" in trad:
            key = "Hindu / Mimamsa"
        elif "Unknown" in trad:
            key = "Unknown / unattributed"
        else:
            key = trad[:30]
        tradition_counts[key] += 1

    print(f"\n  Tradition composition at {label}:")
    for trad, count in sorted(tradition_counts.items(),
                              key=lambda x: -x[1]):
        pct = count / sub.number_of_nodes() * 100
        bar = "█" * int(pct / 2)
        print(f"    {trad:<35} {count:>5} ({pct:>5.1f}%) {bar}")

    print(f"{'='*72}")
    return bc, sub

# ---------------------------------------------------------------------------
# Compare static vs temporal top-N
# ---------------------------------------------------------------------------

def compare_rankings(G, top_n=25):
    print(f"\n{'='*90}")
    print(f"  STATIC vs TEMPORAL ATTRIBUTION — Top {top_n} by Betweenness")
    print(f"  ! = Tier 1 mismatch (earliest cited source predates attributed school)")
    print(f"  ~ = Tier 2 (school proxy — concept may be older than date shown)")
    print(f"{'='*90}")
    print("\n  Computing betweenness on full graph...")
    bc = nx.betweenness_centrality(G, normalized=True)
    top = sorted(bc, key=bc.get, reverse=True)[:top_n]

    print(f"\n  {'#':<4} {'Concept':<26} {'Betw':>7}  "
          f"{'Static (corpus)':<28} {'Temporal':<30} {'Gap':>7}")
    print(f"  {'-'*4} {'-'*26} {'-'*7}  {'-'*28} {'-'*30} {'-'*7}")

    for i, concept in enumerate(top, 1):
        b   = bc[concept]
        at  = G.nodes[concept]
        st  = at.get("static_tradition","?")[:26]
        tt  = at.get("temporal_tradition","?")[:28]
        gap = at.get("mismatch_gap", 0)
        tier= at.get("temporal_tier","?")
        flag= " !" if at.get("mismatch") else (" ~" if tier==2 else "")
        gap_str = f"{gap}y{flag}" if gap > 0 else flag.strip()
        print(f"  {i:<4} {concept:<26} {b:>7.4f}  {st:<28} {tt:<30} {gap_str:>7}")

    tier1_mm = sum(1 for n in top if G.nodes[n].get("mismatch"))
    tier2    = sum(1 for n in top if G.nodes[n].get("temporal_tier")==2)
    print(f"\n  Tier 1 mismatches (!): {tier1_mm}/{top_n} "
          f"(cited source predates attributed school by >100 years)")
    print(f"  Tier 2 proxies (~):    {tier2}/{top_n} "
          f"(school founding date used — may understate antiquity)")
    print(f"{'='*90}")

# ---------------------------------------------------------------------------
# All five key era betweenness snapshots
# ---------------------------------------------------------------------------

def era_betweenness_series(G):
    """
    Run betweenness at 5 key dates and show how the
    top concepts and their traditions change over time.
    This is the main publishable table.
    """
    eras = [
        (-600, "Pre-Buddhist (600 BCE)"),
        (-300, "Early Buddhist/Jain (300 BCE)"),
        ( 300, "Mahayana/Nagarjuna era (300 CE)"),
        ( 800, "Post-Shankara (800 CE)"),
        (1500, "Bhakti movement (1500 CE)"),
    ]

    print(f"\n{'='*72}")
    print("  BETWEENNESS CENTRALITY ACROSS ERAS")
    print("  Top concept and its tradition at each key historical moment")
    print(f"{'='*72}")
    print(f"\n  {'Era':<35} {'Nodes':>6} {'Top concept':<22} "
          f"{'Betw':>7}  Tradition")
    print(f"  {'-'*35} {'-'*6} {'-'*22} {'-'*7}  {'-'*30}")

    for era_ce, label in eras:
        sub = build_era_subgraph(G, era_ce)
        if sub.number_of_nodes() < 5:
            print(f"  {label:<35} {'<5':>6}  (too few nodes)")
            continue
        bc  = nx.betweenness_centrality(sub, normalized=True)
        top = max(bc, key=bc.get)
        trad= sub.nodes[top].get("temporal_tradition","?")[:28]
        tier= sub.nodes[top].get("temporal_tier","?")
        tier_flag = "T1" if tier==1 else "T2"
        print(f"  {label:<35} {sub.number_of_nodes():>6} "
              f"{top:<22} {bc[top]:>7.4f}  {trad} [{tier_flag}]")

    print(f"\n  T1 = explicit scholarly citation")
    print(f"  T2 = school proxy date (lower bound)")
    print(f"{'='*72}")

# ---------------------------------------------------------------------------
# Save graph
# ---------------------------------------------------------------------------

def save_graph(G, path="darshana_temporal_v2.graphml"):
    G_out = G.copy()
    for node, attrs in G_out.nodes(data=True):
        for k,v in list(attrs.items()):
            if v is None:
                G_out.nodes[node][k] = ""
            elif isinstance(v, bool):
                G_out.nodes[node][k] = str(v)
    nx.write_graphml(G_out, path)
    print(f"\n  Saved: {path}")
    print(f"  Open in Gephi: File > Open > {path}")
    print(f"  Colour nodes by: temporal_tradition")
    print(f"  Size nodes by:   betweenness (compute in Gephi: Network > "
          f"Network Analysis > Betweenness Centrality)")
    print(f"  Filter by era:   temporal_date <= your chosen year")
    print(f"\n  Node attributes embedded:")
    attrs = ["static_school","static_tradition","static_date",
             "temporal_date","temporal_tradition","temporal_text",
             "temporal_certainty","temporal_tier",
             "mismatch","mismatch_gap","corpus_count"]
    for a in attrs:
        print(f"    {a}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Temporal graph integration v2 — full coverage"
    )
    parser.add_argument("--era", type=int, default=None)
    parser.add_argument("--compare_rankings", action="store_true")
    parser.add_argument("--era_betweenness", action="store_true")
    parser.add_argument("--save_graph", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--top_n", type=int, default=25)
    args = parser.parse_args()

    print("\n" + "="*72)
    print("  Darshana Temporal Graph v2")
    print("  Three-tier date assignment: citation / school proxy / unknown")
    print("="*72 + "\n")

    # Load
    temporal_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "temporal_source_layer.json"
    )
    timeline = load_temporal_layer(temporal_path)
    print(f"  Tier 1 (explicit citations): {len(timeline)} concept-datings")

    rows = load_darshana(max_rows=args.max_rows)

    print("\n  Building three-tier enriched graph...")
    G = build_enriched_graph(rows, timeline)
    print(f"\n  Graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    if args.era is not None:
        era_snapshot(G, args.era, top_n=args.top_n)
    elif args.compare_rankings:
        compare_rankings(G, top_n=args.top_n)
    elif args.era_betweenness:
        era_betweenness_series(G)
    elif args.save_graph:
        save_graph(G)
    else:
        # Default: compare rankings + era series + three snapshots
        compare_rankings(G, top_n=args.top_n)
        era_betweenness_series(G)
        for era in [-300, 300, 800]:
            era_snapshot(G, era, top_n=10)
        print(f"\n  Run --save_graph to export graphml for Gephi")
        print(f"  Run --era YEAR for a detailed snapshot at any year")

if __name__ == "__main__":
    main()
