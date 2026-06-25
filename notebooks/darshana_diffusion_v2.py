"""
darshana_diffusion_v2.py  —  STANDALONE, no local imports

Philosophical Concept Diffusion across Hindu/Buddhist/Jain Traditions
Based on: Krishnan (2025) Ch5 ABM | Data: joyboseroy/darshana-graph

KEY CHANGE from v1:
  Nodes = individual philosophical CONCEPTS (atman, brahman, sunyata...)
  Edges = typed relations between concepts (IS_IDENTICAL_TO, NEGATES, etc.)
  Node color = school/tradition of origin

  This gives ~2000+ nodes, 28k edges — enough for centrality measures to
  actually diverge across strategies (betweenness vs degree vs percolation).

  The research question becomes:
    If a new syncretic concept enters at a high-betweenness bridge concept
    (e.g. "brahman" which appears in both Hindu and Buddhist discourse),
    does it spread further/faster than seeding at the most-cited concept?

Usage:
  python3 darshana_diffusion_v2.py --seed_concept brahman --steps 30
  python3 darshana_diffusion_v2.py --compare --steps 20 --max_nodes 300
  python3 darshana_diffusion_v2.py --top_concepts   # show bridge concepts
"""

import argparse, os, json, random, urllib.request
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    "purva_mimamsa": "Hindu",    "achintya_bhedabheda": "Hindu",
    "madhyamaka": "Buddhist",    "yogacara": "Buddhist",
    "theravada": "Buddhist",     "vajrayana": "Buddhist",
    "zen": "Buddhist",           "jainism": "Jain",
    "carvaka": "Heterodox",      "ajivika": "Heterodox",
}

TRADITION_COLORS = {
    "Hindu": "#F4A261", "Buddhist": "#2A9D8F",
    "Jain": "#E63946",  "Heterodox": "#ADB5BD", "Mixed": "#9B72CF",
}

STRATEGY_COLORS = {
    "betweenness": "#E63946", "percolation": "#F4A261",
    "degree": "#2A9D8F",      "random": "#ADB5BD",
}

STAGE_LABELS = {0: "Unaware", 1: "Aware", 2: "Action", 3: "Habit"}

# ---------------------------------------------------------------------------
# BeliefAgent (inlined)
# ---------------------------------------------------------------------------

class BeliefAgent:
    def __init__(self, agent_id, persona="neutral", tradition="Mixed"):
        self.id = agent_id
        self.persona = persona
        self.tradition = tradition
        self.intrinsic_benefit = random.uniform(0.2, 0.8)
        self.belief = 0.0
        self.stage = 0
        self.behavior = -1
        self.memory = []
        params = {
            "champion": (0.85, 0.75),
            "skeptic":  (0.20, 0.20),
            "neutral":  (0.50, 0.50),
            "laggard":  (0.15, 0.35),
        }
        self.openness, self.conformity = params.get(persona, (0.5, 0.5))

    def receive(self, belief, stage):
        self.memory.append((belief, stage))
        if len(self.memory) > 6:
            self.memory.pop(0)

    def update(self, cost=0.20):
        if not self.memory:
            return
        avg = sum(b for b, _ in self.memory) / len(self.memory)
        peer = self.conformity * (avg - self.belief)
        delta = self.openness * ((self.intrinsic_benefit - cost) + peer) * 0.25
        self.belief = max(0.0, min(1.0, self.belief + delta))
        if   self.belief > 0.75: self.stage = 3
        elif self.belief > 0.50: self.stage = max(self.stage, 2)
        elif self.belief > 0.25: self.stage = max(self.stage, 1)
        self.behavior = 1 if self.belief > 0.5 else -1

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_edges(max_rows=None):
    try:
        print("  Downloading darshana_graph.jsonl from HuggingFace...")
        rows = []
        with urllib.request.urlopen(HUGGINGFACE_URL, timeout=30) as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                rows.append(json.loads(line.decode("utf-8")))
        print(f"  Loaded {len(rows):,} edges.")
        return rows
    except Exception as e:
        print(f"  Download failed ({e}). Exiting — re-run with internet access.")
        raise SystemExit(1)

# ---------------------------------------------------------------------------
# 2. Build CONCEPT-level graph
# ---------------------------------------------------------------------------
# Add this in build_concept_graph() before building edges
import unicodedata
def normalise_concept(s):
    s = unicodedata.normalize("NFC", s.strip().lower())
    s = s.replace("ṛ", "r").replace("ā", "a").replace("ī", "i").replace("ū", "u")
    s = s.replace("ś", "s").replace("ṣ", "s").replace("ṭ", "t").replace("ḍ", "d")
    s = s.replace("ṇ", "n").replace("ñ", "n").replace("ḥ", "h").replace("ṃ", "m")
    return s

def build_concept_graph(
    rows: List[dict],
    max_nodes: Optional[int] = None,
    min_edge_weight: int = 1,
) -> Tuple[nx.Graph, Dict[str, str]]:
    """
    Nodes = concepts (concept_a, concept_b).
    Edges = co-occurrence in a relation, weighted by frequency.
    concept_school = majority school for each concept node.
    """
    # Filter dirty schools
    rows = [r for r in rows if r.get("school", "") not in DIRTY_SCHOOLS]

    # Track which schools each concept belongs to
    concept_school_counts = defaultdict(lambda: defaultdict(int))
    edge_counts = defaultdict(int)

    for row in rows:
        ca = normalise_concept(row["concept_a"])
        cb = normalise_concept(row["concept_b"])
        school = row.get("school", "").strip().lower()
        if not ca or not cb or ca == cb:
            continue
        concept_school_counts[ca][school] += 1
        concept_school_counts[cb][school] += 1
        key = (min(ca, cb), max(ca, cb))
        edge_counts[key] += 1

    # Majority school per concept
    concept_school = {}
    for concept, school_counts in concept_school_counts.items():
        concept_school[concept] = max(school_counts, key=school_counts.get)

    # Build graph — optionally limit to top-N concepts by degree
    G = nx.Graph()
    for (ca, cb), weight in edge_counts.items():
        if weight >= min_edge_weight:
            G.add_edge(ca, cb, weight=weight)

    # If max_nodes set, keep only the top-N by degree (core subgraph)
    if max_nodes and G.number_of_nodes() > max_nodes:
        top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = {n for n, _ in top}
        G = G.subgraph(keep).copy()
        concept_school = {k: v for k, v in concept_school.items() if k in keep}

    print(f"\n  Concept graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G, concept_school

# ---------------------------------------------------------------------------
# 3. Centralities
# ---------------------------------------------------------------------------

def compute_centralities(G):
    print("  Computing centralities (may take a moment on large graphs)...")
    bc = nx.betweenness_centrality(G, normalized=True)
    dc = nx.degree_centrality(G)
    cc = nx.clustering(G)
    perc_raw = {n: bc[n] * (1 - cc.get(n, 0)) for n in G.nodes()}
    max_p = max(perc_raw.values()) if perc_raw else 1.0
    perc = {k: v / max_p for k, v in perc_raw.items()} if max_p > 0 else perc_raw
    return {"betweenness": bc, "degree": dc, "percolation": perc,
            "random": {n: random.random() for n in G.nodes()}}

def get_seeds(centralities, strategy, G, seed_fraction=0.05):
    k = max(1, int(G.number_of_nodes() * seed_fraction))
    scores = centralities.get(strategy, {})
    return sorted(G.nodes(), key=lambda n: scores.get(n, 0), reverse=True)[:k]

# ---------------------------------------------------------------------------
# 4. Diffusion
# ---------------------------------------------------------------------------

def run_diffusion(G, concept_school, centralities, strategy="betweenness",
                  seed_concept=None, steps=25, cost=0.20, seed_fraction=0.05,
                  rng_seed=42):
    random.seed(rng_seed)
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    agents = {}
    for concept in nodes:
        school = concept_school.get(concept, "")
        tradition = TRADITION_MAP.get(school, "Mixed")
        persona = ("champion" if tradition == "Buddhist"
                   else "skeptic" if tradition == "Heterodox"
                   else "neutral")
        agents[concept] = BeliefAgent(agent_id=concept, persona=persona,
                                      tradition=tradition)

    if seed_concept and seed_concept in agents:
        seeds = [seed_concept]
    else:
        seeds = get_seeds(centralities, strategy, G, seed_fraction)

    print(f"\n  Strategy: {strategy} | Top seeds: {seeds[:5]}{'...' if len(seeds)>5 else ''}")
    for s in seeds:
        if s in agents:
            agents[s].belief = 0.90
            agents[s].stage = 2
            agents[s].behavior = 1

    history = []
    stage_history = []

    for step in range(steps):
        for concept in G.nodes():
            for nbr in G.neighbors(concept):
                w = G[concept][nbr].get("weight", 1)
                influence = agents[concept].belief * min(1.0, w / 5.0)
                agents[nbr].receive(influence, agents[concept].stage)
        for concept in G.nodes():
            agents[concept].update(cost=cost)

        adopters = sum(1 for a in agents.values() if a.behavior == 1)
        rate = adopters / num_nodes
        sc = {STAGE_LABELS[s]: 0 for s in range(4)}
        for a in agents.values():
            sc[STAGE_LABELS[a.stage]] += 1
        history.append(rate)
        stage_history.append(sc)

        bar = "█" * int(rate * 30)
        print(f"  Step {step+1:02d} | {bar:<30} {rate:.0%} | "
              f"Habit:{sc['Habit']:3d} Aware:{sc['Aware']:3d}")

    return agents, history, stage_history, seeds

# ---------------------------------------------------------------------------
# 5. Visualizations
# ---------------------------------------------------------------------------

def plot_comparison(results, steps, output_dir="."):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    for strategy, history in results.items():
        final = history[-1] * 100
        ax.plot(range(1, steps + 1), [h * 100 for h in history],
                label=f"{strategy.capitalize()} (final: {final:.0f}%)",
                color=STRATEGY_COLORS.get(strategy, "#fff"),
                linewidth=2.5, marker="o", markersize=3)

    ax.set_xlabel("Timestep", color="#adb5bd", fontsize=11)
    ax.set_ylabel("Adoption Rate (%)", color="#adb5bd", fontsize=11)
    ax.set_title(
        "Concept Diffusion by Seeding Strategy — Darshana Concept Graph\n"
        "(joyboseroy/darshana-graph | Krishnan 2025 Ch5 ABM methodology)",
        color="#f8f9fa", fontsize=12, pad=12)
    ax.tick_params(colors="#adb5bd")
    ax.spines[:].set_color("#2d3142")
    ax.grid(color="#2d3142", linewidth=0.5, linestyle="--")
    ax.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#f8f9fa", fontsize=10)

    path = os.path.join(output_dir, "darshana_v2_strategy_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {path}")


def plot_network(G, agents, concept_school, seeds, output_dir=".", max_plot_nodes=150):
    # Subsample for plotting if graph is large
    if G.number_of_nodes() > max_plot_nodes:
        # Keep seeds + their neighbors + highest-belief nodes
        seed_set = set(seeds)
        neighbors = set()
        for s in seeds:
            if s in G:
                neighbors.update(G.neighbors(s))
        top_belief = sorted(agents, key=lambda c: agents[c].belief, reverse=True)[:50]
        keep = seed_set | neighbors | set(top_belief)
        keep = list(keep)[:max_plot_nodes]
        G_plot = G.subgraph(keep).copy()
    else:
        G_plot = G

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    pos = nx.spring_layout(G_plot, seed=42, k=0.8)
    max_w = max((G_plot[u][v].get("weight", 1) for u, v in G_plot.edges()), default=1)
    widths = [G_plot[u][v].get("weight", 1) / max_w * 2 for u, v in G_plot.edges()]
    nx.draw_networkx_edges(G_plot, pos, ax=ax, alpha=0.15,
                           edge_color="#5c6370", width=widths)

    for concept in G_plot.nodes():
        school = concept_school.get(concept, "")
        tradition = TRADITION_MAP.get(school, "Mixed")
        color = TRADITION_COLORS.get(tradition, "#9B72CF")
        belief = agents[concept].belief if concept in agents else 0.0
        size = 80 + belief * 400
        border = "#FFD700" if concept in seeds else "#2d3142"
        lw = 2.5 if concept in seeds else 0.8
        nx.draw_networkx_nodes(G_plot, pos, nodelist=[concept], ax=ax,
                               node_color=[color], node_size=[size],
                               edgecolors=[border], linewidths=lw)

    # Label only high-belief or seed nodes
    label_nodes = {c: c for c in G_plot.nodes()
                   if c in seeds or (concept in agents and agents[c].belief > 0.4)}
    if label_nodes:
        nx.draw_networkx_labels(G_plot, pos, labels=label_nodes,
                                ax=ax, font_size=6, font_color="#e9ecef")

    patches = [mpatches.Patch(color=v, label=k) for k, v in TRADITION_COLORS.items()]
    patches.append(mpatches.Patch(color="#FFD700", label="Seed concept"))
    ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#2d3142",
              labelcolor="#f8f9fa", fontsize=9, loc="lower left")
    ax.set_title(
        "Darshana Concept Graph — Belief Diffusion (node size = final belief)\n"
        f"Showing top {G_plot.number_of_nodes()} concepts",
        color="#f8f9fa", fontsize=12, pad=10)
    ax.axis("off")

    path = os.path.join(output_dir, "darshana_v2_network.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_top_concepts(G, centralities, concept_school, n=20):
    bc = centralities["betweenness"]
    dc = centralities["degree"]
    print(f"\n{'='*70}")
    print(f"  {'Concept':<25} {'School':<20} {'Betweenness':>12} {'Degree':>8}")
    print(f"{'='*70}")
    top = sorted(bc, key=bc.get, reverse=True)[:n]
    for concept in top:
        school = concept_school.get(concept, "?")
        tradition = TRADITION_MAP.get(school, "Mixed")
        print(f"  {concept:<25} {school:<20} {bc[concept]:>12.4f} {dc[concept]:>8.4f}  [{tradition}]")
    print(f"{'='*70}")


def print_adoption_summary(agents, concept_school, centralities, top_n=15):
    bc = centralities["betweenness"]
    print(f"\n{'='*72}")
    print(f"  {'Concept':<25} {'School':<18} {'Belief':>8} {'Stage':<10} {'Betw':>7}")
    print(f"{'='*72}")
    ranked = sorted(
        [(c, a) for c, a in agents.items() if a.belief > 0],
        key=lambda x: -x[1].belief
    )[:top_n]
    for concept, agent in ranked:
        school = concept_school.get(concept, "?")
        bar = "█" * int(agent.belief * 10)
        bval = bc.get(concept, 0.0)
        print(f"  {concept:<25} {school:<18} {bar:<8} {STAGE_LABELS[agent.stage]:<10} {bval:>7.4f}")
    print(f"{'='*72}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Darshana Concept Diffusion v2 — concept-level graph"
    )
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Limit JSONL rows (default: all 28k)")
    parser.add_argument("--max_nodes", type=int, default=None,
                        help="Subsample to top-N concepts by degree (default: all)")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cost", type=float, default=0.20)
    parser.add_argument("--seed_fraction", type=float, default=0.05)
    parser.add_argument("--seed_concept", type=str, default=None,
                        help="Seed a specific concept (e.g. brahman, sunyata, atman)")
    parser.add_argument("--strategy", default="betweenness",
                        choices=["betweenness", "percolation", "degree", "random"])
    parser.add_argument("--compare", action="store_true",
                        help="Run all 4 strategies and compare")
    parser.add_argument("--top_concepts", action="store_true",
                        help="Print top bridge concepts by betweenness and exit")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Darshana Diffusion v2 — Concept-Level Graph")
    print("  Data: joyboseroy/darshana-graph (28,322 edges)")
    print("  Nodes: philosophical concepts | Edges: typed relations")
    print("  Method: Krishnan (2025) Ch5 ABM")
    print("="*65)

    rows = load_edges(max_rows=args.max_rows)
    G, concept_school = build_concept_graph(rows, max_nodes=args.max_nodes)
    centralities = compute_centralities(G)

    if args.top_concepts:
        print_top_concepts(G, centralities, concept_school, n=25)
        return

    if args.compare:
        results = {}
        for strategy in ["betweenness", "percolation", "degree", "random"]:
            print(f"\n--- Strategy: {strategy} ---")
            agents, history, _, seeds = run_diffusion(
                G, concept_school, centralities,
                strategy=strategy, steps=args.steps, cost=args.cost,
                seed_fraction=args.seed_fraction,
            )
            results[strategy] = history

        plot_comparison(results, args.steps, args.output_dir)
        print("\nFinal adoption rates:")
        for s, h in sorted(results.items(), key=lambda x: -x[1][-1]):
            print(f"  {s:12s}: {h[-1]:.1%}")
    else:
        agents, history, stage_history, seeds = run_diffusion(
            G, concept_school, centralities,
            strategy=args.strategy,
            seed_concept=args.seed_concept,
            steps=args.steps, cost=args.cost,
            seed_fraction=args.seed_fraction,
        )
        print_adoption_summary(agents, concept_school, centralities)
        plot_network(G, agents, concept_school, seeds, args.output_dir)


if __name__ == "__main__":
    main()
